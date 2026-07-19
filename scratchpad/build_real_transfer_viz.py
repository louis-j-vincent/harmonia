"""build_real_transfer_viz.py — Step 5 deliverable: renders
real_transfer_results.json (Steps 2/3/4's criteria transferred to the 3
real-audio songs, no GT) into a self-contained static HTML file, following
the established '/debug/*' pattern (pre-built static HTML served straight
off disk, no server-side templating — see docs/known_issues.md's earlier
/debug/ssm, /debug/metric-artifact entries for why: it can't drift from
what was actually reviewed).

Shows 3 rows per song: (a) V-measure-optimal tau clustering (the
best-case/ceiling number), (b) Step 2's low-FP-gated tau (the literal
deployment-priority operating point), (c) Step 6's per-song adaptive-
percentile fix — plus the intro-detector verdict and a docked audio player,
so a human can listen and visually compare all three at once.
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent

AUDIO_MAP = {
    "autumn_leaves": "autumn_leaves.m4a",
    "abba_chiquitita": "abba_chiquitita_official_lyric_video.m4a",
    "aretha_chain_of_fools": "aretha_franklin_chain_of_fools_official_lyric_video.m4a",
}

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
          "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
          "#e6beff", "#9a6324", "#800000", "#aaffc3", "#808000",
          "#ffd8b1", "#000075", "#808080", "#ffe119", "#42d4f4"]


def seg_divs(runs, total_bars):
    out = []
    for r in runs:
        width_pct = 100.0 * (r["bar_end"] - r["bar_start"]) / max(1, total_bars)
        lab_num = r["label_id"]
        color = COLORS[lab_num % len(COLORS)]
        out.append(
            '<div class="seg" style="width:%.3f%%;background:%s" '
            'title="%s bars %d-%d">%s</div>' % (
                width_pct, color, r["label"], r["bar_start"], r["bar_end"], r["label"]))
    return "".join(out)


def main():
    data = json.loads((OUT_DIR / "real_transfer_results.json").read_text())
    resplit_path = OUT_DIR / "error_analysis_recursive_split_results.json"
    resplit_data = json.loads(resplit_path.read_text()) if resplit_path.exists() else {}
    fp_path = OUT_DIR / "full_pipeline_eval_results.json"
    fp = json.loads(fp_path.read_text()) if fp_path.exists() else None
    blocks = []
    for song, res in data.items():
        n_bars = res["n_bars"]
        audio_file = AUDIO_MAP.get(song, "")
        audio_tag = ('<audio controls preload="none" src="/audio/%s"></audio>' % audio_file
                     if audio_file else "<em>(no audio mapped)</em>")
        intro = res["intro"]
        intro_badge = ('<span class="badge badge-yes">INTRO DETECTED</span>'
                       if intro["predicted_intro"] else
                       '<span class="badge badge-no">no intro flagged</span>')
        rows = []
        for key, title in [("section_vmeasure_optimal", "(a) V-measure-optimal tau (ceiling, iReal-tuned)"),
                            ("section_low_fp", "(b) Step 2 low-FP tau, target_fpr=0.05 (old deployment priority)"),
                            ("section_fpr010", "(e) NEW: FPR=0.10 interior-optimum tau=0.7759 (iReal's BEST fixed-tau, still fails on real audio)"),
                            ("section_adaptive", "(c) Step 6 FIX: per-song adaptive percentile")]:
            lvl = res[key]
            rows.append(
                '<div class="level-label">%s &middot; tau=%.4f &middot; %d sections</div>'
                '<div class="timeline">%s</div>' % (
                    title, lvl["tau"], lvl["n_sections"], seg_divs(lvl["runs"], n_bars)))
        if song in resplit_data:
            after = resplit_data[song]["after"]
            before = resplit_data[song]["before"]
            rows.append(
                '<div class="level-label">(d) NEW this call: (c) + recursive LOCAL re-split '
                'within long runs (RECOMMENDED) &middot; %d sections &middot; '
                'max run %d&rarr;%d bars &middot; %d runs locally re-split</div>'
                '<div class="timeline">%s</div>' % (
                    after["n_sections"], before["max_run_bars"], after["max_run_bars"],
                    after["n_runs_resplit"], seg_divs(after["runs"], n_bars)))
        blocks.append("""
        <section class="song">
          <h2>%s %s</h2>
          <div class="meta">tempo=%.1f bpm &middot; n_bars=%d &middot; est_tonic_pc=%d
            &middot; intro score=%.3f (thr=%.3f) &middot; offdiag sim mean=%.3f min=%.3f p90=%.3f</div>
          %s
          %s
        </section>
        """ % (song, intro_badge, res["tempo_bpm"], n_bars, res["est_tonic_pc"],
               intro["score"] or 0.0, intro["threshold"], res["offdiag_sanity"]["mean"],
               res["offdiag_sanity"]["min"], res["offdiag_sanity"]["p90"],
               audio_tag, "".join(rows)))

    html = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Merge-criterion real-audio transfer (no GT)</title>
<style>
  body { font-family: -apple-system, sans-serif; margin: 0; padding: 16px;
         background: #111; color: #eee; }
  h1 { font-size: 1.1rem; }
  h2 { font-size: 1rem; margin-bottom: 4px; }
  .note { color: #aaa; font-size: 0.85rem; margin-bottom: 20px; max-width: 900px; }
  .song { margin-bottom: 32px; border-bottom: 1px solid #333; padding-bottom: 16px; }
  .meta { color: #999; font-size: 0.78rem; margin-bottom: 8px; }
  audio { width: 100%; margin-bottom: 10px; }
  .level-label { color: #999; font-size: 0.72rem; margin: 8px 0 2px; }
  .timeline { display: flex; width: 100%; height: 28px; border-radius: 4px;
              overflow: hidden; margin-bottom: 4px; }
  .seg { display: flex; align-items: center; justify-content: center;
         font-size: 0.6rem; color: #000; overflow: hidden;
         white-space: nowrap; border-right: 1px solid rgba(0,0,0,0.3); }
  .badge { font-size: 0.7rem; padding: 2px 8px; border-radius: 10px; margin-left: 8px; }
  .badge-yes { background: #6a2; color: #000; }
  .badge-no { background: #444; color: #aaa; }
  .recbox { background: #1a3a1a; border: 1px solid #3a6a3a; border-radius: 6px;
            padding: 12px 16px; margin-bottom: 20px; max-width: 900px; font-size: 0.85rem; }
  .recbox b { color: #8f8; }
</style></head><body>
<h1>Structure detection: Steps 2/3/4's criteria transferred to real audio (no GT)</h1>
<div class="recbox">
__RECBOX__
</div>
<div class="note">
No section ground truth exists for real audio in this repo — NOT scored,
inspect by eye/ear against the docked audio. Three section-clustering rows
per song: (a) the V-measure-OPTIMAL threshold tuned on clean iReal (ceiling
number, ties flat block8 at 0.682 corpus-wide on iReal) — shown here mainly
as a FAILURE CASE: it collapses aretha_chain_of_fools to 1 section
(over-merges everything), because real audio's off-diagonal similarity
floor sits well above iReal's. (b) Step 2's literal low-false-positive
deployment threshold (tau=0.973, chosen to minimize wrong merges on iReal)
&mdash; the opposite failure: on real audio it barely merges anything
(41/41 blocks stay separate on autumn_leaves). (c) Step 6's fix: a
PER-SONG adaptive threshold (90th percentile of that song's own
off-diagonal similarity distribution, instead of one fixed global iReal
constant) &mdash; non-degenerate on all 3 songs. (d) NEW this call
(error-analysis follow-up on (c)'s remaining failure mode): (c) still
produces very long single-cluster runs on autumn_leaves/abba_chiquitita
(e.g. 80 contiguous bars all one label &mdash; almost certainly several
chorus repeats collapsed together, since Autumn Leaves is a 32-bar form).
Row (d) locally re-clusters any run &gt;=32 bars at a stricter LOCAL
percentile. Result is a genuine but PARTIAL improvement: autumn_leaves
12&rarr;16 sections, abba_chiquitita 10&rarr;14 sections, longest run on
abba drops 48&rarr;40 bars &mdash; but autumn_leaves still has one
untouched 80-bar run (its local distribution didn't clear the local
threshold either, i.e. that specific span really is harmonically flat at
this similarity resolution). Stated honestly: NOT a complete fix, a
real incremental one. Full reasoning: docs/known_issues.md, search "Step 6",
"real_transfer", and "recursive" (or "error-analysis"), dated 2026-07-18.
Same color anywhere within one row = predicted same group at that level;
colors are NOT comparable across rows or songs.
</div>
__BLOCKS__
</body></html>""".replace("__BLOCKS__", "".join(blocks))

    if fp:
        s = fp["summary"]
        recbox = (
            "<b>RECOMMENDED PIPELINE (consolidation call, 2026-07-18):</b> "
            "on iReal, where ground truth exists, the full pipeline (Step 3 intro-trim "
            "+ Follow-up 2's FPR=0.10 interior-optimum bar-merge criterion, tau=0.7759) "
            "scores V_F=%.4f&plusmn;%.4f corpus-scale (5 seeds, %d tunes) vs flat block8's "
            "V_F=%.4f&plusmn;%.4f &mdash; a +%.4f delta, i.e. a <b>statistical TIE</b>, not a "
            "reliable win (matches the established pattern: chord-only similarity structure "
            "detectors tie block8, they don't beat it). "
            "On REAL AUDIO (no GT, below), that SAME FPR=0.10 threshold (row (e)) behaves "
            "almost identically to the already-known over-merge failure (tau=0.78): it "
            "collapses aretha_chain_of_fools to 1 section and only reaches 4 sections on "
            "autumn_leaves. <b>The per-song adaptive-percentile fix + recursive local "
            "re-split (row (d)) remains the deployed real-audio recommendation</b> &mdash; "
            "the better iReal threshold does NOT reduce how much that patch is needed; "
            "if anything it makes the patch MORE necessary, since 0.7759 sits even closer "
            "to the collapse point than the old 0.973 low-FP threshold was. "
            "Full numbers: scratchpad/full_pipeline_eval_results.json, "
            "scratchpad/real_transfer_results.json." % (
                s["full_pipeline_VF_mean"], s["full_pipeline_VF_std"], fp["n_corpus"],
                s["block8_VF_mean"], s["block8_VF_std"], s["delta_full_vs_block8"]))
    else:
        recbox = "(full_pipeline_eval_results.json not found — run scratchpad/full_pipeline_eval.py)"
    html = html.replace("__RECBOX__", recbox)

    outp = OUT_DIR / "real_transfer_viz.html"
    outp.write_text(html)
    print("wrote", outp)


if __name__ == "__main__":
    main()
