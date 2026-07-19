"""Build the static HTML report for the boundary-bleed verification."""
import json
from pathlib import Path

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
OUT = REPO / "docs/bleed_verification_2026_07_16"
ex = json.loads((OUT / "examples.json").read_text())
PC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SAMPLE_MS = 1000.0 / 44100.0  # one-sample tolerance

max_err = max(abs(e["clip_err_ms"]) for e in ex)
max_post = max(e["post_bleed_ms"] for e in ex)
max_pre = max(e["pre_bleed_ms"] for e in ex)
fr_ms = ex[0]["fr_period_ms"]

cards = []
for e in ex:
    gt = PC[e["gt_root"]] if e["gt_root"] is not None else "?"
    nxt = PC[e["next_root"]] if e["next_root"] is not None else "?"
    err_ok = abs(e["clip_err_ms"]) < SAMPLE_MS
    bleed_ok = e["post_bleed_ms"] == 0.0 and e["pre_bleed_ms"] == 0.0
    cards.append(f"""
  <div class="card">
    <div class="card-head">
      <span class="status">{e['label']}</span>
      <span class="song">{e['song']} #{e['idx']}</span>
    </div>
    <div class="note">{e['note']}</div>
    <div class="labels">
      <div class="label-row"><span class="k">GT span [t0,t1)</span>
        <span class="v">[{e['t0']:.3f}, {e['t1']:.3f}]  ({e['span']:.3f}s)</span></div>
      <div class="label-row"><span class="k">GT root</span>
        <span class="v gt">{gt}</span>
        <span class="k">next chord</span>
        <span class="v pred">{e['next_label']} (root {nxt}) @ {e['next_t']:.3f}s</span></div>
      <div class="label-row"><span class="k">frames pooled</span>
        <span class="v">{e['n_frames']} @ 86.13 Hz{' [MIN_FRAMES floor]' if e['floored'] else ''}</span></div>
      <div class="label-row"><span class="k">pooled window</span>
        <span class="v">[{e['w0']:.3f}, {e['w1']:.3f}] (frame centres)</span></div>
    </div>
    <div class="metrics">
      <span class="chip {'ok' if bleed_ok else 'bad'}">PRE-bleed {e['pre_bleed_ms']:.2f} ms</span>
      <span class="chip {'ok' if bleed_ok else 'bad'}">POST-bleed {e['post_bleed_ms']:.2f} ms</span>
      <span class="chip {'ok' if err_ok else 'bad'}">clip {e['clip_dur']:.4f}s (Δ {e['clip_err_ms']:+.3f} ms)</span>
    </div>
    <img class="chroma" src="chroma/{e['id']}.png" alt="temporal chroma {e['id']}">
    <audio controls preload="none" src="clips/{e['id']}.wav"></audio>
    <div class="meta">audio = EXACT [t0,t1), zero padding &middot; verify: this is precisely the frame set the model pools</div>
  </div>""")

html = f"""<title>Boundary-Bleed Verification — is contamination really 0.0 ms?</title>
<style>
  :root {{ --cream:#f7f3e9; --maroon:#8a2b2b; --ink:#2b2622; --ok:#2c6e49; --bad:#8a2b2b; }}
  * {{ box-sizing:border-box; }}
  body {{ background:var(--cream); color:var(--ink); font-family:Georgia,'Times New Roman',serif;
    max-width:1200px; margin:0 auto; padding:2rem 1.5rem 4rem; }}
  h1 {{ font-style:italic; font-weight:normal; font-size:1.9rem; margin-bottom:0.2rem; }}
  .subtitle {{ color:#665; margin-bottom:1.2rem; font-size:0.95rem; }}
  .intro {{ background:#fffef9; border:1px solid #ddd3bd; border-left:4px solid var(--maroon);
    padding:0.9rem 1.2rem; margin-bottom:1.5rem; font-size:0.92rem; line-height:1.55; }}
  .intro code {{ font-family:Menlo,monospace; font-size:0.85em; background:#efe9d9; padding:0 0.25em; }}
  .verdict {{ background:#eef6ef; border:1px solid #bcd8c2; border-left:4px solid var(--ok);
    padding:0.9rem 1.2rem; margin-bottom:2rem; font-size:0.92rem; line-height:1.55; }}
  .verdict.caveat {{ background:#fbf4e6; border-color:#e2cfa0; border-left-color:#b8860b; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(420px,1fr)); gap:1rem; }}
  .card {{ background:#fffef9; border:1px solid #ddd3bd; border-radius:6px; padding:0.9rem;
    display:flex; flex-direction:column; gap:0.5rem; }}
  .card-head {{ display:flex; justify-content:space-between; align-items:baseline; }}
  .status {{ font-weight:bold; font-family:Menlo,monospace; color:var(--maroon); font-size:0.95rem; }}
  .song {{ color:#776; font-size:0.8rem; }}
  .note {{ font-size:0.82rem; color:#665; font-style:italic; }}
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
  img.chroma {{ width:100%; border:1px solid #e5ddc8; border-radius:4px; }}
  audio {{ width:100%; }}
  .meta {{ font-size:0.68rem; color:#aa9; font-family:Menlo,monospace; }}
  .legend {{ font-size:0.82rem; color:#665; margin:1rem 0 1.5rem; line-height:1.5; }}
  .legend b.green {{ color:#2c8c3a; }} .legend b.red {{ color:var(--bad); }}
  .legend b.yel {{ color:#b8860b; }}
</style>

<h1>Boundary-Bleed Verification — is contamination really 0.0 ms?</h1>
<div class="subtitle">Direct audible + visual test of the frame-clip pooling fix, on the EXACT [t0,t1) the model pools</div>

<div class="intro">
  <b>Why this exists.</b> The fix to <code>scripts/build_rwc_corpus.py::build_song</code> (frame-clip
  pooling, replacing beat-grid snapping) claimed the ~310 ms mean / 476 ms max next-chord feature
  contamination went to <b>0.0 ms</b>. The earlier root-error tool showed clips with <b>±0.25 s padding</b>
  (for listening comfort) — which is exactly what made the bleed ambiguous. So here <b>every audio clip is
  the exact, unpadded <code>[t0, t1)</code> window</b>, and the chroma shown is the exact per-frame
  (86.13 Hz) activation set the fixed pooler selects — reusing <code>build_song</code>'s own
  <code>searchsorted(ft,·,'left')</code> + <code>MIN_FRAMES</code> logic verbatim, not an approximation.
  <br><br>
  <b>How to read each chroma plot.</b> All 4 BP48 blocks (onset / note / bass / treble), pitch-class on y,
  time on x. A <b class="green">green box</b> marks the <b>exact frames the model pools</b> (the feature is
  their sum). The plot deliberately shows <b>extra context on both sides</b> of the box so you can see what is
  <i>excluded</i>. <b class="red">Red line</b> = span end t1; <b class="yel">dashed yellow</b> = next-chord
  onset; dotted horizontal lines = GT root (green) and next-chord root (yellow). If the fix works, the next
  chord's chroma should appear only <b>to the right of the box</b>, never inside it.
</div>

<div class="verdict">
  <b>Result (all 10 examples).</b> Every pooled window's frame centres lie strictly inside <code>[t0,t1)</code>:
  PRE-bleed max <b>{max_pre:.2f} ms</b>, POST-bleed max <b>{max_post:.2f} ms</b> — the "0.0 ms" claim holds
  at frame-selection granularity. Every audio clip's measured duration matches <code>t1−t0</code> to within
  <b>{max_err:.3f} ms</b> (one 44.1 kHz sample = {SAMPLE_MS:.4f} ms), verified by <code>ffprobe</code>, not
  eyeballed. The MIN_FRAMES=4 floor did not fire on any example (shortest span 0.433 s = 37 frames).
</div>

<div class="verdict caveat">
  <b>Honest residual (does NOT contradict 0.0 ms, but state it).</b> "0.0 ms" means <b>zero next-chord
  frames</b> are pooled. Each Basic Pitch frame is ~{fr_ms:.1f} ms wide, so the last in-span frame still
  integrates audio out to ~½ frame (~{fr_ms/2:.1f} ms) past its centre — an irreducible sub-frame tail at
  86.13 Hz, versus the old <b>310 ms mean / 476 ms</b>. That is ~50–80× smaller than one chord and far below
  audible/harmonic relevance. The box edge is drawn at the last pooled frame's <i>centre</i>, so a hair of
  visual space between the box and the red t1 line is expected, not bleed.
</div>

<div class="legend">
  Listen to each clip while watching its chroma: the audio you hear is <b>byte-for-byte the span the feature
  is pooled over</b>. On the smoking-gun case (<b>RWC_P091 #21, C:7 → F:min</b>) the old model predicted
  <b class="red">F</b> = the next chord's root; in the bass block the <b>F</b> energy now sits entirely to the
  right of the green box, i.e. excluded from the feature.
</div>

<div class="grid">
{''.join(cards)}
</div>
"""
(OUT / "index.html").write_text(html)
print(f"Wrote {OUT/'index.html'} ({len(html)} bytes), {len(ex)} cards")
