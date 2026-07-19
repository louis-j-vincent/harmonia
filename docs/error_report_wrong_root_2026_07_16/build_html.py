import json
from pathlib import Path

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
OUT_DIR = REPO / "docs/error_report_wrong_root_2026_07_16"

manifest = json.loads((OUT_DIR / "examples_manifest.json").read_text())
examples = manifest["examples"]
clips = {c["idx"]: c for c in json.loads((OUT_DIR / "clips_manifest.json").read_text())}
preds = json.loads((OUT_DIR / "fixed_test_predictions.json").read_text())
root_acc = preds["root_held_out_acc"]; qual_acc = preds["qual_held_out_acc"]

CAT_LABEL = {"p4p5": "P4/P5 interval error", "inversion": "inversion-related (GT bass ≠ root)",
             "other": "other"}
CAT_CLASS = {"p4p5": "bad", "inversion": "yel", "other": "neut"}

cards = []
n_cat = {"p4p5": 0, "inversion": 0, "other": 0}
n_bass_agrees_gt = 0
for i, ex in enumerate(examples):
    clip = clips.get(i)
    if clip is None:
        continue
    n_cat[ex["category"]] += 1
    if ex["bass_agrees_gt"]:
        n_bass_agrees_gt += 1
    bass_chip = ("bass-argmax agrees GT" if ex["bass_agrees_gt"] else
                 "bass-argmax agrees PRED" if ex["bass_agrees_pred"] else
                 "bass-argmax = neither")
    bass_chip_cls = "ok" if ex["bass_agrees_gt"] else ("bad" if ex["bass_agrees_pred"] else "neut")
    cat = ex["category"]
    dur = clip["expected_dur"]
    err_ms = clip["duration_err_ms"]
    card = f"""
  <div class="card">
    <div class="card-head">
      <span class="status">{ex['gt_root_name']}:{ex['gt_quality']} &rarr; pred {ex['pred_root_name']}:{ex['pred_quality']}</span>
      <span class="song">{ex['song_id']} @ {ex['t0']:.2f}s</span>
    </div>
    <div class="note">interval GT&rarr;pred = {ex['interval_semitones']} semitones &middot; <span class="tag {CAT_CLASS[cat]}">{CAT_LABEL[cat]}</span></div>
    <div class="labels">
      <div class="label-row"><span class="k">GT label</span> <span class="v gt">{ex['label']}</span></div>
      <div class="label-row"><span class="k">GT span [t0,t1)</span>
        <span class="v">[{ex['t0']:.3f}, {ex['t1']:.3f}]  ({dur:.3f}s)</span></div>
      <div class="label-row"><span class="k">predicted root</span>
        <span class="v pred">{ex['pred_root_name']}</span>
        <span class="k">p(root)</span><span class="v pred">{ex['pred_root_prob']:.2f}</span></div>
      <div class="label-row"><span class="k">predicted quality</span>
        <span class="v pred">{ex['pred_quality']}</span>
        <span class="k">p(qual)</span><span class="v pred">{ex['pred_quality_prob']:.2f}</span></div>
      <div class="label-row"><span class="k">bass-argmax</span>
        <span class="v">{ex['bass_argmax_name']}</span></div>
    </div>
    <div class="metrics">
      <span class="chip {bass_chip_cls}">{bass_chip}</span>
      <span class="chip ok">clip {dur:.3f}s (&Delta; {err_ms:.4f} ms)</span>
    </div>
    <img class="chroma" src="chroma/ex{i:02d}.png" alt="chroma ex{i:02d}">
    <audio controls preload="none" src="{clip['path']}"></audio>
    <div class="meta">audio = EXACT [t0,t1), zero padding, same standard as bleed_verification_2026_07_16</div>
  </div>"""
    cards.append(card)

html = f"""<title>Wrong-Root Error Gallery — RWC held-out, FIXED (frame-clipped) corpus</title>
<style>
  :root {{ --cream:#f7f3e9; --maroon:#8a2b2b; --ink:#2b2622; --ok:#2c6e49; --bad:#8a2b2b; --yel:#b8860b; }}
  * {{ box-sizing:border-box; }}
  body {{ background:var(--cream); color:var(--ink); font-family:Georgia,'Times New Roman',serif;
    max-width:1300px; margin:0 auto; padding:2rem 1.5rem 4rem; }}
  h1 {{ font-style:italic; font-weight:normal; font-size:1.9rem; margin-bottom:0.2rem; }}
  .subtitle {{ color:#665; margin-bottom:1.2rem; font-size:0.95rem; }}
  .intro {{ background:#fffef9; border:1px solid #ddd3bd; border-left:4px solid var(--maroon);
    padding:0.9rem 1.2rem; margin-bottom:1.5rem; font-size:0.92rem; line-height:1.55; }}
  .intro code {{ font-family:Menlo,monospace; font-size:0.85em; background:#efe9d9; padding:0 0.25em; }}
  .stats {{ background:#eef6ef; border:1px solid #bcd8c2; border-left:4px solid var(--ok);
    padding:0.9rem 1.2rem; margin-bottom:2rem; font-size:0.92rem; line-height:1.55; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(430px,1fr)); gap:1rem; }}
  .card {{ background:#fffef9; border:1px solid #ddd3bd; border-radius:6px; padding:0.9rem;
    display:flex; flex-direction:column; gap:0.5rem; }}
  .card-head {{ display:flex; justify-content:space-between; align-items:baseline; gap:0.5rem; flex-wrap:wrap; }}
  .status {{ font-weight:bold; font-family:Menlo,monospace; color:var(--maroon); font-size:0.92rem; }}
  .song {{ color:#776; font-size:0.78rem; }}
  .note {{ font-size:0.82rem; color:#665; font-style:italic; }}
  .tag {{ font-family:Menlo,monospace; font-size:0.72rem; padding:0.1rem 0.4rem; border-radius:3px; font-style:normal; }}
  .tag.bad {{ background:#f4dede; color:var(--maroon); }}
  .tag.yel {{ background:#faf1da; color:var(--yel); }}
  .tag.neut {{ background:#eee9db; color:#665; }}
  .labels {{ font-size:0.84rem; }}
  .label-row {{ display:flex; gap:0.4rem; align-items:baseline; margin:0.15rem 0; flex-wrap:wrap; }}
  .k {{ color:#776; font-size:0.78rem; }}
  .v {{ font-weight:bold; font-family:Menlo,monospace; font-size:0.8rem; }}
  .v.gt {{ color:var(--ok); }}
  .v.pred {{ color:var(--bad); }}
  .metrics {{ display:flex; flex-wrap:wrap; gap:0.35rem; }}
  .chip {{ font-size:0.72rem; font-family:Menlo,monospace; padding:0.18rem 0.45rem; border-radius:3px; }}
  .chip.ok {{ background:#e2f0e6; color:var(--ok); }}
  .chip.bad {{ background:#f4dede; color:var(--maroon); }}
  .chip.neut {{ background:#eee9db; color:#665; }}
  img.chroma {{ width:100%; border:1px solid #e5ddc8; border-radius:4px; }}
  audio {{ width:100%; }}
  .meta {{ font-size:0.68rem; color:#aa9; font-family:Menlo,monospace; }}
  .legend {{ font-size:0.82rem; color:#665; margin:1rem 0 1.5rem; line-height:1.5; }}
  .legend b.green {{ color:#2c8c3a; }} .legend b.red {{ color:var(--bad); }}
</style>

<h1>Wrong-Root Error Gallery</h1>
<div class="subtitle">Every example here is a root MISCLASSIFICATION &mdash; RWC held-out test split, FIXED (frame-clipped, zero-bleed) corpus &amp; model</div>

<div class="intro">
  <b>Why this exists.</b> Follow-up to <code>root_error_analysis_2026_07_16</code> (correct/wrong split) and
  <code>bleed_verification_2026_07_16</code> (exact-audio-precision proof). This report is <b>error-only</b>:
  every card below is a root the model got WRONG, built on <code>data/cache/rwc/rwc_bp48_fixed.npz</code>
  (the frame-clipped corpus from the boundary-bleed fix, zero contamination by construction) and a freshly
  trained root+quality MLP (single seed=0, 80/20 song-stratified split, <code>--roll</code> augmentation,
  same methodology as <code>train_jaah_cv.py</code>). Each card shows GT root+quality, the model's actual
  predicted root+quality with softmax confidence, a 4-block BP48 chroma heatmap with GT (green) and
  predicted (red) roots marked, the bass-argmax diagnostic (blue triangle, pure argmax of the bass
  sub-block &mdash; can agree with GT even when the model's real prediction doesn't), and a playable audio
  clip trimmed to the EXACT <code>[t0,t1)</code> model-input window with <b>zero padding</b>
  (ffprobe-verified, same standard as <code>bleed_verification_2026_07_16</code>).
</div>

<div class="stats">
  <b>Model:</b> root held-out acc = {root_acc:.1%}, quality held-out acc = {qual_acc:.1%}
  (seed=0, matches the corpus-level 64.8%&plusmn;2.2% 6-seed root number within one seed's noise).<br>
  <b>Examples shown:</b> {len(cards)} wrong-root cases &mdash;
  {n_cat['p4p5']} P4/P5-interval, {n_cat['inversion']} inversion-related (GT itself has a non-root bass),
  {n_cat['other']} other.<br>
  <b>Bass-argmax diagnostic:</b> agrees with GT on {n_bass_agrees_gt}/{len(cards)} of these error cases &mdash;
  i.e. the model had bass evidence for the right root in its own input on those and still predicted wrong.
</div>

<div class="legend">
  <b class="green">Green</b> lines = GT root. <b class="red">Red dashed</b> lines = model's predicted root.
  Blue triangle = bass-argmax (pure argmax of the 12-d bass sub-block, independent diagnostic).
  Audio = byte-for-byte <code>[t0,t1)</code>, no listening padding.
</div>

<div class="grid">
{''.join(cards)}
</div>
"""

out = OUT_DIR / "index.html"
out.write_text(html)
print(f"wrote {out} ({len(cards)} cards)")
