"""scripts/tune_lock_propagation.py — simulated-lock ROI harness + operating
point for the lock-propagation feature (branch feat/chord-context-prior,
task 3 of the final-integration brief; design: docs/design_chord_context_prior.md
evaluation ladder step 3).

WHAT THIS MEASURES
------------------
For each eval song: decode the DISPLAYED chart via the exact production path
(``infer_chords_v1`` with the live nnls24 config, same as ``_run_analysis`` /
the Brick-0 scorer's ``SHIPPED_CONFIG``), align it to independent chord
ground truth, then simulate a user locking ONE span at a time (to GT) and
call the SAME rescore core the ``/api/context_rescore`` endpoint calls --
``span_rescore.differential_rescore`` over ``span_rescore.compute_acoustic_logp``'s
acoustic evidence. Two lock scenarios, scored separately:

  * lock a WRONG span (displayed != GT) to GT — the realistic "user fixes an
    error" action.  ROI_diff = mean number of OTHER spans that flip wrong ->
    correct AS A RESULT OF THE LOCK (see below).
  * lock an already-CORRECT span (displayed == GT, sampled) — must not
    corrupt neighbours.  CORR_diff = mean number of OTHER spans that flip
    correct -> wrong as a result of the lock.

DIFFERENTIAL METRICS (2026-07-30 redesign, replacing this file's first
version): the first version of this harness diffed the locked rescore's
output directly against the DISPLAYED chart. That conflates lock-caused
changes with the rescore's own unconditional disagreement with the displayed
chart, which is large and present even with ZERO locks (churn 15-30% of
spans on real songs, because a repeated progression fragment gets the same
reading everywhere its trigram context recurs, independent of any lock) --
diagnosed directly on blue_bossa: locking ONE span changed 64/189 spans vs
displayed, but only 8 of those 64 differ from a ZERO-lock call at the SAME
(lam, delta, K). ``differential_rescore`` fixes this the same way the OLD,
WORKING ``/api/reinfer`` always has: decode a baseline (no constraints) and a
locked run (with constraints) under IDENTICAL settings, and diff locked vs
baseline, never vs the original display. NO-LOCK churn is therefore zero BY
CONSTRUCTION now (baseline vs baseline) and is dropped as a swept metric/
constraint (task 2's problem is solved structurally by the redesign, not by
tuning delta down its effect).

The (lambda, delta) sweep is cheap PURE NUMPY once the expensive parts —
production decode (~network-free, cache-hit acoustic features) and posterior
pooling — are computed ONCE per song and reused for every config. This script
does exactly that: a decode+alignment pass first (cached to disk), then a
sweep over the cached per-song tensors. K is fixed at 6 (swept once in the
first version of this harness; never moved the ranking near the top, dropped
here per the differential-redesign brief).

INVENTORY (run standalone: ``--inventory-only``)
-------------------------------------------------
Checks which real-audio corpora have (a) independent chord GT, (b) audio on
disk, (c) ``data/cache/musx_probs/<stem>.npz`` cached (never triggers a fresh
musx run silently — a cache MISS just makes ``compute_acoustic_logp`` fall
back to the nnls_heads backend, which is fine, just a different acoustic
source than the app's primary).

USAGE
-----
    .venv/bin/python scripts/tune_lock_propagation.py --inventory-only
    .venv/bin/python scripts/tune_lock_propagation.py            # full run
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.request
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.eval.accuracy_score import SHIPPED_CONFIG, _decode_to_wav  # noqa: E402
from harmonia.models import span_rescore  # noqa: E402
from harmonia.models.chord_context_prior import (  # noqa: E402
    QUAL5,
    load_context_prior,
    parse_harte_lite,
)
from harmonia.models.span_rescore import differential_rescore  # noqa: E402

CACHE_DIR = REPO / "data" / "cache"
WORK_DIR = REPO / "data" / "cache" / "tune_lock_propagation"
WAV_DIR = WORK_DIR / "wav"
CHART_CACHE = WORK_DIR / "charts"
for _d in (WORK_DIR, WAV_DIR, CHART_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

GUITARSET_AUDIO = REPO / "data/cache/guitarset/audio"
GUITARSET_ANNOT = REPO / "data/cache/guitarset/annotation"
ALIGNED_MANIFEST = REPO / "data/chord_dataset/manifest.jsonl"
RWC_AUDIO = REPO / "docs/audio/rwc_rwc_p001.m4a"
RWC_CHORD_URL = ("https://raw.githubusercontent.com/rwc-music/rwc-annotations/"
                  "main/01_annotations_preprocessed/chords/RWC-P/RWC_P001.csv")
JAAH_DIR = REPO / "data/cache/choco/partitions/jaah"


# ═══════════════════════════════════════════════════════════════════════════
# INVENTORY
# ═══════════════════════════════════════════════════════════════════════════

def build_inventory() -> list[dict]:
    """Corpus x (n audio-on-disk, n with independent GT, n with musx_probs
    cached) — printed as a table, also returned for the doc."""
    rows = []

    # GuitarSet: 360 audio+JAMS pairs, real audio on disk.
    gs_audio = sorted(p.stem for p in GUITARSET_AUDIO.glob("*.wav")) if GUITARSET_AUDIO.exists() else []
    gs_annot = {p.stem for p in GUITARSET_ANNOT.glob("*.jams")} if GUITARSET_ANNOT.exists() else set()
    gs_gt = [s for s in gs_audio if s.removesuffix("_mic") in gs_annot]
    gs_musx = [s for s in gs_gt if (CACHE_DIR / "musx_probs" / f"{s}.npz").exists()]
    rows.append({"corpus": "GuitarSet", "n_audio": len(gs_audio), "n_with_gt": len(gs_gt),
                "n_with_musx_probs": len(gs_musx),
                "note": "360 excerpts = 6 players x 5 backing tracks x {comp,solo}; "
                        "chord GT = JAMS 'chord' namespace (simplified maj/min/7/hdim7)"})

    # RWC-Popular: audio GONE except rwc_rwc_p001.m4a (per harmonia/eval/
    # benchmark_set.py "WHAT SURVIVES ON DISK"); its GT is fetchable (small,
    # per-song CSV from GitHub, NOT the 4GB Zenodo audio zip).
    rwc_audio_n = 1 if RWC_AUDIO.exists() else 0
    rwc_musx = 1 if (CACHE_DIR / "musx_probs" / f"{RWC_AUDIO.stem}.npz").exists() else 0
    rows.append({"corpus": "RWC-Popular", "n_audio": rwc_audio_n, "n_with_gt": rwc_audio_n,
                "n_with_musx_probs": rwc_musx,
                "note": "full-song audio licensed/gone for 99/100 tracks (verified: "
                        "data/cache/rwc/audio/RWC-P/ is an empty dir tree); only "
                        "RWC_P001's audio survives (docs/audio/rwc_rwc_p001.m4a). "
                        "GT fetched live from github.com/rwc-music/rwc-annotations "
                        "(small per-song CSV, not the big Zenodo zip)."})

    # aligned chord_dataset: 139 records / 8 unique songs, all audio verified
    # present on disk (manifest's own audio_path field, re-checked here).
    n_records = n_songs = n_on_disk = n_musx = 0
    if ALIGNED_MANIFEST.exists():
        by_song = defaultdict(set)
        with open(ALIGNED_MANIFEST) as f:
            for line in f:
                r = json.loads(line)
                n_records += 1
                by_song[r["song_id"]].add(r["audio_path"])
        n_songs = len(by_song)
        for sid, paths in by_song.items():
            (p,) = paths  # verified 1:1 song_id -> audio_path earlier
            if (REPO / p).exists():
                n_on_disk += 1
                if (CACHE_DIR / "musx_probs" / f"{Path(p).stem}.npz").exists():
                    n_musx += 1
    rows.append({"corpus": "aligned_corpus (chart<->audio alignment output)",
                "n_audio": n_on_disk, "n_with_gt": n_records,
                "n_with_musx_probs": n_musx,
                "note": f"{n_records} scored segments / {n_songs} unique songs (same "
                        "audio referenced by multiple segments); GT timestamps are "
                        "SELF-DERIVED from the alignment process (benchmark_set.py's "
                        "GT-provenance table: 'labels OK, at self-derived timestamps "
                        "— NO for beat/alignment eval, circular'), acceptable here "
                        "since we only need approximate label-at-time, not exact "
                        "onset precision."})

    # JAAH via ChoCo: labels only, verified zero audio anywhere on disk.
    jaah_jams = len(list((JAAH_DIR / "choco" / "jams").glob("*.jams"))) if JAAH_DIR.exists() else 0
    rows.append({"corpus": "JAAH (via ChoCo)", "n_audio": 0, "n_with_gt": jaah_jams,
                "n_with_musx_probs": 0,
                "note": "labels-only: verified zero audio files anywhere under data/ "
                        "for any jaah_*.jams id — NOT usable for this harness "
                        "(needs production-path DECODE, which needs audio)."})

    return rows


def print_inventory(rows: list[dict]) -> None:
    print("\n=== INVENTORY (corpus x usable-song counts) ===")
    print(f"{'corpus':<45} {'n_audio':>8} {'n_with_gt':>10} {'n_musx_probs':>13}")
    for r in rows:
        print(f"{r['corpus']:<45} {r['n_audio']:>8} {r['n_with_gt']:>10} {r['n_with_musx_probs']:>13}")
    print()
    for r in rows:
        print(f"- {r['corpus']}: {r['note']}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# EVAL SET (bounded: ~19 songs across 3 real-audio corpora, per the "cap at
# ~15-20 songs, balanced across corpora" time-budget rule)
# ═══════════════════════════════════════════════════════════════════════════

# GuitarSet: 10 of the 30 stems that ALREADY have musx_probs cached (no
# backfill needed) -- 2 players (00, 03) x 5 backing tracks, all "_comp_mic"
# (comping = chordal guitar, acoustically matched to the chord GT; "_solo_mic"
# stems are single-note improvisation over the same GT progression and would
# be a much weaker acoustic signal).
GUITARSET_STEMS = [
    f"{player}_{tune}_comp_mic"
    for player in ("00", "03")
    for tune in ("BN1-129-Eb", "Funk1-114-Ab", "Jazz1-130-D", "Rock1-130-A", "SS1-100-C#")
]

# aligned_corpus: all 8 songs (all verified on disk with musx_probs cached).
ALIGNED_SONG_IDS = [
    "bein_green", "blue_bossa", "blue_bossa_backing", "close_to_you",
    "every_breath_you_take", "georgia_on_my_mind", "let_it_be", "stand_by_me",
]


def resolve_aligned_audio() -> dict[str, Path]:
    by_song = {}
    with open(ALIGNED_MANIFEST) as f:
        for line in f:
            r = json.loads(line)
            by_song.setdefault(r["song_id"], REPO / r["audio_path"])
    return by_song


# ═══════════════════════════════════════════════════════════════════════════
# GT loaders — each returns a sorted list of (t0, t1, root_pc, qual5_idx)
# ═══════════════════════════════════════════════════════════════════════════

def gt_events_guitarset(stem: str) -> list[tuple[float, float, int, int]]:
    annot_stem = stem.removesuffix("_mic")
    jams = json.loads((GUITARSET_ANNOT / f"{annot_stem}.jams").read_text())
    chord_anns = [a for a in jams["annotations"] if a["namespace"] == "chord"]
    # The FIRST chord annotation is the simplified maj/min/7/hdim7 chart-level
    # GT (verified above: clean 4-value vocabulary, no extensions/bass) --
    # closer to "what chord was this" than the second (voicing-level, with
    # extensions + explicit bass degree) transcription.
    data = chord_anns[0]["data"]
    out = []
    for d in data:
        parsed = parse_harte_lite(d["value"])
        if parsed is None:
            continue
        t0, t1 = d["time"], d["time"] + d["duration"]
        out.append((t0, t1, parsed[0], parsed[1]))
    return sorted(out)


def gt_events_aligned(song_id: str) -> list[tuple[float, float, int, int]]:
    out = []
    with open(ALIGNED_MANIFEST) as f:
        for line in f:
            r = json.loads(line)
            if r["song_id"] != song_id:
                continue
            for c in r["chords"]:
                parsed = parse_harte_lite(c["label"])
                if parsed is None:
                    continue
                out.append((c["t0"], c["t1"], parsed[0], parsed[1]))
    return sorted(out)


def fetch_rwc_p001_gt() -> list[tuple[float, float, int, int]]:
    req = urllib.request.Request(RWC_CHORD_URL,
                                 headers={"User-Agent": "harmonia-research/1.0 "
                                                        "(louisjvincent@gmail.com)"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        text = resp.read().decode("utf-8")
    rd = csv.reader(io.StringIO(text), delimiter=";")
    next(rd, None)  # header
    out = []
    for row in rd:
        if len(row) != 3:
            continue
        t0, t1, label = float(row[0]), float(row[1]), row[2].strip()
        parsed = parse_harte_lite(label)
        if parsed is None:
            continue
        out.append((t0, t1, parsed[0], parsed[1]))
    return sorted(out)


def gt_at(events: list[tuple[float, float, int, int]], t: float) -> tuple[int, int] | None:
    """Linear scan (event lists here are at most a few hundred long — fine)."""
    for t0, t1, root, q5 in events:
        if t0 <= t < t1:
            return (root, q5)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Production-path decode (cached to disk: expensive, computed ONCE per song)
# ═══════════════════════════════════════════════════════════════════════════

def get_displayed_chart(name: str, audio_path: Path) -> list[dict]:
    """[{start_s, end_s, label}, ...] via the SHIPPED production pipeline
    (harmonia.eval.accuracy_score.SHIPPED_CONFIG == the live nnls24 config
    _run_analysis uses). Cached to CHART_CACHE/<name>.json."""
    cache_f = CHART_CACHE / f"{name}.json"
    if cache_f.exists():
        return json.loads(cache_f.read_text())
    from harmonia.models.chord_pipeline_v1 import infer_chords_v1

    wav = _decode_to_wav(audio_path, WAV_DIR)
    t0 = time.time()
    chart = infer_chords_v1(wav, cache_dir=CACHE_DIR, **SHIPPED_CONFIG)
    elapsed = time.time() - t0
    chords = [{"start_s": c["start_s"], "end_s": c["end_s"], "label": c["label"]}
             for c in chart.chords if c["end_s"] > c["start_s"]]
    cache_f.write_text(json.dumps({"chords": chords, "decode_s": elapsed,
                                   "wav_stem": wav.stem}))
    print(f"  decoded {name}: {len(chords)} chords, {elapsed:.1f}s")
    return {"chords": chords, "decode_s": elapsed, "wav_stem": wav.stem}


def load_displayed_chart(name: str) -> dict:
    cache_f = CHART_CACHE / f"{name}.json"
    return json.loads(cache_f.read_text())


# ═══════════════════════════════════════════════════════════════════════════
# Per-song precompute: displayed tokens, GT tokens (or None), acoustic logp.
# The ONLY per-config-varying step downstream is lattice_rescore (pure numpy).
# ═══════════════════════════════════════════════════════════════════════════

def precompute_song(name: str, audio_path: Path,
                    gt_events: list[tuple[float, float, int, int]]) -> dict | None:
    raw = get_displayed_chart(name, audio_path)
    chords = raw["chords"]
    wav = WAV_DIR / f"{raw['wav_stem']}.wav"

    spans, displayed, labels = [], [], []
    for c in chords:
        parsed = parse_harte_lite(c["label"])
        if parsed is None:      # shouldn't happen for a real chart, but be defensive
            continue
        spans.append((c["start_s"], c["end_s"]))
        displayed.append(parsed)
        labels.append(c["label"])
    if not spans:
        return None

    gt = [gt_at(gt_events, 0.5 * (t0 + t1)) for t0, t1 in spans]

    result = span_rescore.compute_acoustic_logp(wav, spans, fallback_audio=wav)
    return {
        "name": name, "spans": spans, "displayed": displayed, "labels": labels,
        "gt": gt, "acoustic_logp": result["logp"], "backend": result["backend"],
        "cache_hit": result["cache_hit"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Sweep: for a fixed (lam, delta, K), score every song's simulated locks.
# ═══════════════════════════════════════════════════════════════════════════

def _diff_counts(displayed, gt, chosen, skip_idx, n):
    """One lock's effect on every OTHER GT-scored span: (exact_fixed,
    exact_corrupted, root_fixed, root_corrupted, qual_fixed, qual_corrupted)."""
    ef = ec = rf = rc = qf = qc = 0
    for j in range(n):
        if j == skip_idx or gt[j] is None:
            continue
        d, g, c = displayed[j], gt[j], chosen[j]
        if c != d:
            if c == g and d != g:
                ef += 1
            if d == g and c != g:
                ec += 1
        if c[0] != d[0]:
            if c[0] == g[0] and d[0] != g[0]:
                rf += 1
            if d[0] == g[0] and c[0] != g[0]:
                rc += 1
        if c[1] != d[1]:
            if c[1] == g[1] and d[1] != g[1]:
                qf += 1
            if d[1] == g[1] and c[1] != g[1]:
                qc += 1
    return ef, ec, rf, rc, qf, qc


def score_config(songs: list[dict], context_scorer, lam: float, delta: float, K: int,
                 collect_examples: bool = False, margin_gate: float = 0.0) -> dict:
    """Differential metrics (2026-07-30 redesign): every lock is scored via
    ``differential_rescore``, which itself runs baseline (no locks) + locked
    at the identical (lam, delta, K) and returns ONLY lock-attributable
    changes (see that function's docstring). No-lock churn is dropped: it is
    zero by construction under this scheme (baseline vs baseline), not a
    swept quantity anymore.

    ``margin_gate`` (2026-07-30, "Margin gate" re-analysis): screens
    propagated (non-locked) changes by how confidently the locked run prefers
    its own answer over the baseline's -- see ``differential_rescore``'s own
    docstring for the exact per-span comparison. 0.0 (default) reproduces the
    section-8 behaviour exactly.
    """
    agg = defaultdict(float)
    examples = []
    for s in songs:
        n = len(s["displayed"])
        acoustic = s["acoustic_logp"]
        displayed, gt = s["displayed"], s["gt"]

        # Bounded lock samples per song (time-budget guardrail: per-call DP cost
        # scales with n, so with 19 songs incl. two 100-190-span ones, an
        # UNCAPPED wrong-lock count made an early, unstaged version of this
        # sweep project past 30 minutes -- trimmed here, deterministic stride
        # sampling so results are reproducible, not random).
        wrong_idxs_all = [i for i in range(n) if gt[i] is not None and displayed[i] != gt[i]]
        w_stride = max(1, len(wrong_idxs_all) // 10)
        wrong_idxs = wrong_idxs_all[::w_stride][:10]
        correct_idxs_all = [i for i in range(n) if gt[i] is not None and displayed[i] == gt[i]]
        c_stride = max(1, len(correct_idxs_all) // 6)
        correct_idxs = correct_idxs_all[::c_stride][:6]

        for i in wrong_idxs:
            locks = [None] * n
            locks[i] = gt[i]
            chosen, changed, margins = differential_rescore(
                acoustic, displayed, locks, context_scorer, lam=lam, K=K, delta=delta,
                margin_gate=margin_gate)
            ef, ec, rf, rc, qf, qc = _diff_counts(displayed, gt, chosen, i, n)
            agg["wrong_locks"] += 1
            agg["wrong_exact_fixed"] += ef
            agg["wrong_exact_corrupted"] += ec
            agg["wrong_root_fixed"] += rf
            agg["wrong_root_corrupted"] += rc
            agg["wrong_qual_fixed"] += qf
            agg["wrong_qual_corrupted"] += qc
            if collect_examples and ef > 0:
                examples.append({"song": s["name"], "lock_idx": i, "kind": "wrong->gt",
                                 "spans": s["spans"], "labels": s["labels"],
                                 "displayed": displayed, "gt": gt, "chosen": chosen,
                                 "fixed": ef, "corrupted": ec})

        for i in correct_idxs:
            locks = [None] * n
            locks[i] = gt[i]
            chosen, changed, margins = differential_rescore(
                acoustic, displayed, locks, context_scorer, lam=lam, K=K, delta=delta,
                margin_gate=margin_gate)
            ef, ec, rf, rc, qf, qc = _diff_counts(displayed, gt, chosen, i, n)
            agg["correct_locks"] += 1
            agg["correct_exact_fixed"] += ef
            agg["correct_exact_corrupted"] += ec
            agg["correct_root_corrupted"] += rc
            agg["correct_qual_corrupted"] += qc
            if collect_examples and ec > 0:
                examples.append({"song": s["name"], "lock_idx": i, "kind": "correct-lock-corrupted",
                                 "spans": s["spans"], "labels": s["labels"],
                                 "displayed": displayed, "gt": gt, "chosen": chosen,
                                 "fixed": ef, "corrupted": ec})

    n_wrong = max(agg["wrong_locks"], 1)
    n_correct = max(agg["correct_locks"], 1)
    n_all_locks = max(agg["wrong_locks"] + agg["correct_locks"], 1)
    out = {
        "lam": lam, "delta": delta, "K": K,
        # roi_diff / corr_diff: DIFFERENTIAL metrics (differential_rescore
        # already restricts `chosen` to lock-attributable changes only, so
        # these counts can no longer include pre-existing baseline churn --
        # see the module docstring's "DIFFERENTIAL METRICS" section).
        "roi_diff": agg["wrong_exact_fixed"] / n_wrong,
        "roi_diff_root": agg["wrong_root_fixed"] / n_wrong,
        "roi_diff_qual": agg["wrong_qual_fixed"] / n_wrong,
        "corr_diff_wrong_locks": agg["wrong_exact_corrupted"] / n_wrong,
        "corr_diff_correct_locks": agg["correct_exact_corrupted"] / n_correct,
        "corr_diff_combined": (agg["wrong_exact_corrupted"] + agg["correct_exact_corrupted"]) / n_all_locks,
        "corr_diff_root_correct_locks": agg["correct_root_corrupted"] / n_correct,
        "corr_diff_qual_correct_locks": agg["correct_qual_corrupted"] / n_correct,
        "n_wrong_locks": int(agg["wrong_locks"]), "n_correct_locks": int(agg["correct_locks"]),
    }
    if collect_examples:
        out["examples"] = examples
    return out


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory-only", action="store_true")
    args = ap.parse_args()

    inv_rows = build_inventory()
    print_inventory(inv_rows)
    if args.inventory_only:
        return

    # ── build the eval set ──────────────────────────────────────────────
    eval_songs_spec = []
    for stem in GUITARSET_STEMS:
        eval_songs_spec.append(("guitarset", stem, GUITARSET_AUDIO / f"{stem}.wav",
                                lambda s=stem: gt_events_guitarset(s)))
    aligned_audio = resolve_aligned_audio()
    for sid in ALIGNED_SONG_IDS:
        eval_songs_spec.append(("aligned", sid, aligned_audio[sid],
                                lambda s=sid: gt_events_aligned(s)))
    print("Fetching RWC_P001 chord GT (small CSV, github.com/rwc-music/rwc-annotations)...")
    rwc_gt = fetch_rwc_p001_gt()
    eval_songs_spec.append(("rwc", "rwc_p001", RWC_AUDIO, lambda: rwc_gt))

    print(f"\n=== EVAL SET: {len(eval_songs_spec)} songs "
         f"({sum(1 for c,_,_,_ in eval_songs_spec if c=='guitarset')} GuitarSet, "
         f"{sum(1 for c,_,_,_ in eval_songs_spec if c=='aligned')} aligned_corpus, "
         f"{sum(1 for c,_,_,_ in eval_songs_spec if c=='rwc')} RWC) ===\n")

    # ── decode + precompute (expensive; cached) ─────────────────────────
    songs = []
    for corpus, name, audio_path, gt_fn in eval_songs_spec:
        gt_events = gt_fn()
        t0 = time.time()
        pre = precompute_song(f"{corpus}_{name}", audio_path, gt_events)
        if pre is None:
            print(f"  SKIP {corpus}/{name}: production decode produced 0 usable spans")
            continue
        pre["corpus"] = corpus
        n_gt = sum(1 for g in pre["gt"] if g is not None)
        n_wrong = sum(1 for d, g in zip(pre["displayed"], pre["gt"]) if g is not None and d != g)
        print(f"  {corpus}/{name}: {len(pre['spans'])} spans, {n_gt} GT-scored "
             f"({n_wrong} wrong), backend={pre['backend']} cache_hit={pre['cache_hit']} "
             f"[{time.time()-t0:.1f}s]")
        songs.append(pre)

    # ── spot-check parser against raw labels (verify-don't-trust), 2 per corpus ──
    print("\n=== spot-check: displayed chart labels parse cleanly ===")
    seen_corpus = set()
    for s in songs:
        if s["corpus"] in seen_corpus:
            continue
        seen_corpus.add(s["corpus"])
        for lab, tok in list(zip(s["labels"], s["displayed"]))[:2]:
            print(f"  [{s['corpus']}] {lab!r} -> root_pc={tok[0]} qual5={QUAL5[tok[1]]}")

    # ── the sweep (cheap: pure numpy per config, reusing the precompute) ──
    # DIFFERENTIAL redesign (2026-07-30): K fixed at 6 (swept in the earlier,
    # non-differential version of this harness; never moved the ranking near
    # the top there, dropped per the redesign brief), lam x delta = 16 configs.
    context_scorer = load_context_prior(corpus="pooled")
    LAMS = [0.5, 1, 2, 4]
    DELTAS = [0, 0.5, 1, 2]
    K_FIXED = 6

    print(f"\n=== differential sweep: {len(LAMS)}x{len(DELTAS)} = "
         f"{len(LAMS)*len(DELTAS)} configs at K={K_FIXED}, over {len(songs)} songs ===")
    t_sweep0 = time.time()
    results = []
    for lam, delta in product(LAMS, DELTAS):
        r = score_config(songs, context_scorer, lam, delta, K_FIXED)
        results.append(r)
        print(f"  lam={lam} delta={delta} K={K_FIXED}: roi_diff={r['roi_diff']:.3f} "
             f"corr_diff_wrong={r['corr_diff_wrong_locks']:.3f} "
             f"corr_diff_correct={r['corr_diff_correct_locks']:.3f} "
             f"[{time.time()-t_sweep0:.0f}s elapsed]")
    print(f"sweep done in {time.time()-t_sweep0:.1f}s, {len(results)} configs scored")

    results.sort(key=lambda r: -r["roi_diff"])
    print("\n=== top 10 by ROI_diff ===")
    header = ("lam", "delta", "K", "roi_diff", "roi_root", "roi_qual",
              "corr_wrong", "corr_correct")
    print(" ".join(f"{h:>11}" for h in header))
    for r in results[:10]:
        print(" ".join(f"{r[k]:>11.3f}" if isinstance(r[k], float) else f"{r[k]:>11}"
                       for k in ("lam", "delta", "K", "roi_diff", "roi_diff_root", "roi_diff_qual",
                                "corr_diff_wrong_locks", "corr_diff_correct_locks")))

    # ── pick operating point: max ROI_diff s.t. corr_diff_correct<=0.05/lock
    # AND corr_diff_wrong <= 0.25*roi_diff (net clearly positive on the
    # wrong-lock case; churn is dropped as a constraint -- zero by
    # construction under differential_rescore, see module docstring) ──
    CORRECT_LOCK_CORRUPTION_CAP = 0.05
    WRONG_LOCK_CORRUPTION_RATIO_CAP = 0.25
    feasible = [r for r in results
               if r["corr_diff_correct_locks"] <= CORRECT_LOCK_CORRUPTION_CAP
               and r["corr_diff_wrong_locks"] <= WRONG_LOCK_CORRUPTION_RATIO_CAP * max(r["roi_diff"], 1e-9)]
    feasible.sort(key=lambda r: -r["roi_diff"])
    chosen = feasible[0] if feasible else None

    print(f"\n=== feasible configs (corr_diff_correct<=0.05/lock, "
         f"corr_diff_wrong<=0.25*roi_diff): {len(feasible)}/{len(results)} ===")
    if chosen:
        print(f"CHOSEN: lam={chosen['lam']} delta={chosen['delta']} K={chosen['K']} "
             f"-> roi_diff={chosen['roi_diff']:.3f} "
             f"corr_diff_wrong={chosen['corr_diff_wrong_locks']:.3f} "
             f"corr_diff_correct={chosen['corr_diff_correct_locks']:.3f}")
    else:
        print("NO feasible config under the constraints — reporting best cells honestly.")
        chosen = max(results, key=lambda r: r["roi_diff"])
        print(f"BEST-ROI (infeasible) cell: lam={chosen['lam']} delta={chosen['delta']} "
             f"K={chosen['K']} -> roi_diff={chosen['roi_diff']:.3f} "
             f"corr_diff_combined={chosen['corr_diff_combined']:.3f}")

    # ── concrete examples at the chosen operating point ──────────────────
    print("\n=== collecting concrete examples at the chosen operating point ===")
    detailed = score_config(songs, context_scorer, chosen["lam"], chosen["delta"], chosen["K"],
                           collect_examples=True)
    examples = sorted(detailed.get("examples", []), key=lambda e: -(e["fixed"] - e["corrupted"]))

    out = {
        "inventory": inv_rows,
        "eval_songs": [{"name": s["name"], "corpus": s["corpus"], "n_spans": len(s["spans"]),
                       "n_gt_scored": sum(1 for g in s["gt"] if g is not None),
                       "n_wrong": sum(1 for d, g in zip(s["displayed"], s["gt"])
                                     if g is not None and d != g),
                       "backend": s["backend"], "cache_hit": s["cache_hit"]}
                      for s in songs],
        "sweep": results,
        "chosen": chosen,
        "examples": examples[:8],
    }
    out_path = WORK_DIR / "sweep_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
