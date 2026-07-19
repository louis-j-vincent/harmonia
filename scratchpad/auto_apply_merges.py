"""auto_apply_merges.py — 2026-07-18 (overnight autonomous call, tasks 2+3):
wire AUTO-tier bar-merge candidates (tier=="auto", sim>=TAU_AUTO=0.96) to
apply WITHOUT a human tap, and measure the effect across all 3 real songs.

**DO NOT run this as a default/automatic step or wire it into any live
pipeline path.** Running it (see `main()` below) is safe in the sense that
it never mutates any stored chart — it only measures, via a direct
pipeline call plus a preview-only `/api/reinfer` round-trip — but the
MEASUREMENT ITSELF found tau_auto=0.96 does not transfer to real audio:
61% of touched bars had confidence go DOWN and 36% flipped label when
this script's own auto-tier merges were applied on the 3 real songs (see
`docs/known_issues.md`, "AUTO-tier auto-apply WIRED and MEASURED", and
`scratchpad/realaudio_threshold_check.py`'s corpus-scale corroboration:
pooled baseline-label agreement only 39.4% at tau=0.96, 62.5% at tau=0.99
— nowhere near the 98-99% precision tau_auto's symbolic-iReal-corpus
derivation implied). This script is kept as a validated, working
MECHANISM (the multi-merge batch wiring + the pool_beat_evidence
batch-abort bug it found and fixed are both real, reusable) — not as a
recommended action. Do not build a UI trigger for it or call it from
`/api/analyze` until tau_auto has been recalibrated directly against
real-audio ground truth (see known_issues.md's "NEXT STEP").

**WHERE THIS RUNS (explicit decision, per the brief's options (a)/(b)/(c)):
option (a), a one-time batch/bake step, run from this script.** NOT wired
into `/api/analyze` (option (b) — rejected: bigger blast radius, changes
default pipeline behavior for every future analysis silently, CLAUDE.md
rule #6 territory the brief explicitly said to avoid without strong
justification). A UI-triggered "auto-apply high-confidence merges" action
(option (c)) is left as documented future work, NOT built this call —
the brief said (a) or (c), and (a) is the one that directly produces the
measurement task 3 asks for, reproducibly, without touching UI files that
were only just unlocked and are still sanctioned-rare-edit surfaces.
This script is independently re-runnable (a plain function call / CLI
invocation per song), satisfying the "not a one-off manual action"
requirement.

**Candidate source**: `scratchpad/bar_merge_full_census_<slug>.json` (this
call's own new files, generated via `bar_merge_candidates.candidate_groups
(algo="knn", max_candidates=100000)` — i.e. the FULL k-NN-selected edge
pool for each song, not the UI-facing top-20-capped
`bar_merge_candidates_<slug>.json`). tier=="auto" means sim>=TAU_AUTO=0.96
(unchanged this call; only the SUGGEST floor moved, see
`bar_merge_candidates.py`'s DEFAULT_TAU docstring).

**Pairs -> GROUPS decision (explicit, per the brief's caution about the
pairs-only format / transitive-closure over-merge collapse elsewhere in
this thread — see REFRAME/bakeoff entries in known_issues.md)**: auto-tier
PAIRS are resolved into connected-component GROUPS via union-find, but
ONLY up to `MAX_GROUP_SIZE` bars; any component that would exceed the cap
falls back to its individual constituent PAIRS instead of being closed
transitively. Checked empirically first (this call): the deployed k-NN
generator's own `max_pairs_per_bar=2` dedup already bounds auto-tier
component size to <=6 bars on all 3 real songs (17/16/54 components,
max size 6 — see session log) — nowhere near the 71-183-bar collapse the
naive full-threshold+union-find approach produced elsewhere in this
project. The cap (default 8) is a defensive backstop that is NOT expected
to fire given that finding, not a load-bearing part of the design.

**Measurement path**: a DIRECT PIPELINE CALL (sanctioned by the brief as
an alternative to curl, "not a bypassed simulation") — `infer_chords_v1`
called twice (unconstrained baseline, then with `user_constraints={
"confirms":[], "merges": groups}`), the EXACT same function and same
merges-present code branch `/api/reinfer` itself calls when `merges` is
non-empty (verified by reading `scripts/harmonia_server.py::api_reinfer` —
it explicitly skips the billboard backend and routes to `infer_chords_v1`
whenever merges are present). This is necessary, not just convenient: the
live `/api/reinfer` endpoint's `diff` only reports bars whose LABEL
changed, but task 3 needs before/after CONFIDENCE for every touched bar
even when the label didn't change (pooling can move confidence without
flipping the argmax label) — that data isn't exposed by the HTTP
response shape at all. A real curl round-trip against the live
`/api/reinfer/<file>` endpoint is ALSO run per song (see
`curl_verify_reinfer` below) as an independent check that the production
HTTP path agrees with the direct-call numbers wherever they overlap
(the label-changed subset) — both paths are exercised, not one substituted
for the other.
"""
from __future__ import annotations
import sys, json, shutil, subprocess, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

REPO = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO / "docs" / "audio"
OUT_DIR = Path(__file__).resolve().parent
MAX_GROUP_SIZE = 8

SONGS = {
    "aretha_franklin_chain_of_fools_official_lyric_video": {
        "chart_file": "inferred_aretha_franklin_chain_of_fools_official_lyric_video.html",
        "audio_name": "aretha_franklin_chain_of_fools_official_lyric_video.m4a",
    },
    "autumn_leaves": {
        "chart_file": "inferred_autumn_leaves.html",
        "audio_name": "autumn_leaves.m4a",
    },
    "abba_chiquitita_official_lyric_video": {
        "chart_file": "inferred_abba_chiquitita_official_lyric_video.html",
        "audio_name": "abba_chiquitita_official_lyric_video.m4a",
    },
}


def union_find_groups(pairs):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in pairs:
        union(i, j)
    groups = {}
    for k in parent:
        groups.setdefault(find(k), set()).add(k)
    return [sorted(g) for g in groups.values()]


def build_auto_groups(slug, max_group_size=MAX_GROUP_SIZE):
    path = OUT_DIR / f"bar_merge_full_census_{slug}.json"
    data = json.loads(path.read_text())
    auto_pairs = [tuple(c["bars"]) for c in data["candidates"] if c["tier"] == "auto"]
    bar_span = {}
    for c in data["candidates"]:
        for b, sp in zip(c["bars"], c["spans"]):
            bar_span[b] = sp
    comps = union_find_groups(auto_pairs)
    groups, fell_back = [], []
    for comp in comps:
        if len(comp) > max_group_size:
            fell_back.append(comp)
            for (i, j) in auto_pairs:
                if i in comp and j in comp:
                    groups.append(sorted([i, j]))
        else:
            groups.append(comp)
    return groups, bar_span, fell_back, data["meta"], len(auto_pairs), len(comps)


def transcode(audio_path, wav_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-ac", "1", "-ar", "22050", str(wav_path)],
        check=True, capture_output=True, timeout=180,
    )


def chord_at(chords, t):
    for c in chords:
        if c["start_s"] <= t < c["end_s"]:
            return c
    return None


def run_direct_pipeline(slug, groups, bar_span):
    """Direct call into the exact production code path (see module docstring)."""
    from harmonia.models.chord_pipeline_v1 import infer_chords_v1
    audio_path = AUDIO_DIR / SONGS[slug]["audio_name"]
    tmp = Path(tempfile.mkdtemp(prefix="harmonia_autoapply_"))
    try:
        wav = tmp / "a.wav"
        transcode(audio_path, wav)
        cache = tmp  # shared cache_dir -> 2nd call is a stage-1 cache hit, same as api_reinfer
        base = infer_chords_v1(wav, cache_dir=cache, joint_transition_weight=0.0)
        merges = [{"spans": [bar_span[str(b)] if str(b) in bar_span else bar_span[b] for b in g]}
                  for g in groups]
        cons = infer_chords_v1(wav, cache_dir=cache, joint_transition_weight=0.0,
                                user_constraints={"confirms": [], "merges": merges})
        return base, cons, merges
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def curl_verify_reinfer(chart_file, merges, port=7771):
    """Real HTTP round-trip against the live production server (independent
    check against the direct-call numbers on the label-changed subset)."""
    import urllib.request
    url = f"http://localhost:{port}/api/reinfer/{chart_file}"
    body = json.dumps({"confirms": [], "merges": merges}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def measure_song(slug, do_curl=True):
    groups, bar_span, fell_back, census_meta, n_auto_pairs, n_components = build_auto_groups(slug)
    print(f"\n=== {slug} ===")
    print(f"  auto-tier pairs: {n_auto_pairs}  connected components: {n_components}  "
          f"groups applied: {len(groups)}  fell_back(>{MAX_GROUP_SIZE} bars): {len(fell_back)}")
    all_bars = sorted({b for g in groups for b in g})
    print(f"  distinct bars touched: {len(all_bars)}  group sizes: {sorted(len(g) for g in groups)}")

    t0 = time.time()
    base, cons, merges_payload = run_direct_pipeline(slug, groups, bar_span)
    print(f"  direct-pipeline calls done in {time.time()-t0:.1f}s")

    base_ch = [c for c in base.chords if c["end_s"] > c["start_s"]]
    cons_ch = [c for c in cons.chords if c["end_s"] > c["start_s"]]

    per_bar = []
    for b in all_bars:
        t0b, t1b = bar_span[b]
        mid = 0.5 * (t0b + t1b)
        bc = chord_at(base_ch, mid)
        cc = chord_at(cons_ch, mid)
        if bc is None or cc is None:
            continue
        per_bar.append({
            "bar": b, "t0": t0b, "t1": t1b,
            "before_label": bc["label"], "after_label": cc["label"],
            "before_confidence": bc.get("confidence", 0.0),
            "after_confidence": cc.get("confidence", 0.0),
            "label_changed": bc["label"] != cc["label"],
            "confidence_delta": cc.get("confidence", 0.0) - bc.get("confidence", 0.0),
        })

    n_label_changed = sum(1 for r in per_bar if r["label_changed"])
    deltas = np.array([r["confidence_delta"] for r in per_bar])
    befores = np.array([r["before_confidence"] for r in per_bar])
    afters = np.array([r["after_confidence"] for r in per_bar])
    n_regressed = int(np.sum(deltas < -1e-9))
    regressions = [r for r in per_bar if r["confidence_delta"] < -1e-9]

    print(f"  bars matched (before+after found): {len(per_bar)}")
    print(f"  label changed: {n_label_changed}/{len(per_bar)}")
    print(f"  confidence before: mean={befores.mean():.4f} std={befores.std():.4f}")
    print(f"  confidence after:  mean={afters.mean():.4f} std={afters.std():.4f}")
    print(f"  confidence delta:  mean={deltas.mean():+.4f} std={deltas.std():.4f} "
          f"min={deltas.min():+.4f} max={deltas.max():+.4f}")
    print(f"  REGRESSIONS (confidence went DOWN): {n_regressed}/{len(per_bar)}")
    if regressions:
        for r in regressions[:10]:
            print(f"    bar {r['bar']}: {r['before_label']}@{r['before_confidence']:.4f} -> "
                  f"{r['after_label']}@{r['after_confidence']:.4f}  (delta {r['confidence_delta']:+.4f})")

    curl_result = None
    if do_curl:
        try:
            curl_result = curl_verify_reinfer(SONGS[slug]["chart_file"], merges_payload)
            rejected = curl_result.get("rejected", [])
            n_changed_http = curl_result.get("n_changed")
            print(f"  CURL verify: HTTP 200, n_changed={n_changed_http}, rejected={rejected}")
            # cross-check: every label-changed bar found by the direct call should
            # correspond to a diff entry from the curl round-trip covering that time span
            http_diff_spans = [(d["start_s"], d["end_s"]) for d in curl_result.get("diff", [])]
            mismatches = 0
            for r in per_bar:
                if not r["label_changed"]:
                    continue
                mid = 0.5 * (r["t0"] + r["t1"])
                covered = any(s0 <= mid < s1 for s0, s1 in http_diff_spans)
                if not covered:
                    mismatches += 1
            print(f"  cross-check: {mismatches}/{n_label_changed} direct-call label-changes "
                  f"NOT found in curl diff (expect 0)")
        except Exception as e:
            print(f"  CURL verify FAILED: {e}")
            curl_result = {"error": str(e)}

    return {
        "slug": slug, "n_auto_pairs": n_auto_pairs, "n_components": n_components,
        "n_groups_applied": len(groups), "n_fell_back_components": len(fell_back),
        "group_sizes": sorted(len(g) for g in groups),
        "n_bars_touched": len(per_bar),
        "n_label_changed": n_label_changed,
        "confidence_before_mean": float(befores.mean()) if len(befores) else None,
        "confidence_after_mean": float(afters.mean()) if len(afters) else None,
        "confidence_delta_mean": float(deltas.mean()) if len(deltas) else None,
        "confidence_delta_std": float(deltas.std()) if len(deltas) else None,
        "n_regressions": n_regressed,
        "regressions": regressions,
        "per_bar": per_bar,
        "curl_verify": {k: v for k, v in (curl_result or {}).items() if k != "chords"},
    }


def main():
    results = {}
    for slug in SONGS:
        results[slug] = measure_song(slug)

    all_deltas = np.concatenate([np.array([r["confidence_delta"] for r in results[s]["per_bar"]])
                                  for s in SONGS]) if any(results[s]["per_bar"] for s in SONGS) else np.array([])
    total_bars = sum(results[s]["n_bars_touched"] for s in SONGS)
    total_changed = sum(results[s]["n_label_changed"] for s in SONGS)
    total_regressed = sum(results[s]["n_regressions"] for s in SONGS)
    total_groups = sum(results[s]["n_groups_applied"] for s in SONGS)

    print("\n\n=== AGGREGATE ACROSS ALL 3 SONGS ===")
    print(f"  total groups applied: {total_groups}")
    print(f"  total bars touched: {total_bars}")
    print(f"  total label changes: {total_changed}")
    print(f"  total confidence regressions: {total_regressed}")
    if len(all_deltas):
        print(f"  pooled confidence delta: mean={all_deltas.mean():+.4f} std={all_deltas.std():.4f}")

    out = {"songs": results, "aggregate": {
        "total_groups": total_groups, "total_bars_touched": total_bars,
        "total_label_changes": total_changed, "total_regressions": total_regressed,
        "pooled_confidence_delta_mean": float(all_deltas.mean()) if len(all_deltas) else None,
        "pooled_confidence_delta_std": float(all_deltas.std()) if len(all_deltas) else None,
    }}
    (OUT_DIR / "auto_apply_results.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nwrote scratchpad/auto_apply_results.json")


if __name__ == "__main__":
    main()
