"""Generate real (not fake) draft-vs-final chord data for the progressive-
analysis demo: run infer_chords_v1 twice on the same song — once with pure
NNLS-24 heads (the FAST ~4-6s pass, no music-x-lab) and once with the full
musx-routed config (the ~20-30s "final" pass) — then align both onto the same
bar grid (same beat tracking -> same bt -> same n_bars) so the demo can show
a real draft chord flipping to a real corrected chord, bar by bar.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from harmonia.models.chord_pipeline_v1 import infer_chords_v1
from scripts.render_youtube_chart import chart_to_interactive_inputs
from harmonia.output.chart_model import to_chart_model
from harmonia.output.chart_interactive import render_interactive

AUDIO = Path("/tmp/demo_autumn.wav")
TITLE = "Autumn Leaves"

t0 = time.perf_counter()
draft = infer_chords_v1(AUDIO, feature_frontend="nnls24", bass_frontend="nnls24",
                         quality_frontend="nnls24", segment_source="nnls",
                         beat_period_mode="bestfit", cache_dir=Path("data/cache"))
t_draft = time.perf_counter() - t0

t0 = time.perf_counter()
final = infer_chords_v1(AUDIO, feature_frontend="nnls24", bass_frontend="musx",
                         quality_frontend="musx", segment_source="nnls",
                         beat_period_mode="bestfit", cache_dir=Path("data/cache"))
t_final = time.perf_counter() - t0

print(f"draft pass: {t_draft:.2f}s  ({len(draft.chords)} chords)")
print(f"final pass: {t_final:.2f}s  ({len(final.chords)} chords)")
print(f"tempo={final.tempo_bpm} key={final.global_key} duration={final.duration_s:.1f}s")


def bars_from_chart(chart):
    chart_obj, chord_dicts = chart_to_interactive_inputs(chart, TITLE, "demo", bar1_offset_beats=0)
    render_interactive(chart_obj, chord_dicts, Path("/tmp/_demo_scratch.html"),
                        bars_per_row=4, sections=chart.sections)
    payload_text = Path("/tmp/_demo_scratch.html").read_text()
    import re
    m = re.search(r"^const P = (\{.*\});\s*$", payload_text, re.M)
    payload = json.loads(m.group(1))
    model = to_chart_model(payload, filename="demo.html", title=TITLE)
    return model


draft_model = bars_from_chart(draft)
final_model = bars_from_chart(final)

# Flatten both models to one chord per bar (first chord's root/q in that bar,
# or "held"/empty) so draft and final line up 1:1 by bar index.
NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def flatten(model):
    out = []
    for sec in model["sections"]:
        for bar in sec["bars"]:
            if bar:
                c = bar[0]
                out.append({"root": c["root"], "q": c["q"], "t0": c["t0"]})
            else:
                out.append(None)  # held bar
    return out


draft_bars = flatten(draft_model)
final_bars = flatten(final_model)
n_bars = min(len(draft_bars), len(final_bars))
print(f"draft n_bars={len(draft_bars)} final n_bars={len(final_bars)} using {n_bars}")

bars_out = []
last_draft, last_final = None, None
for i in range(n_bars):
    d, f = draft_bars[i], final_bars[i]
    d_lbl = f"{NOTE[d['root']]}{d['q']}" if d else None
    f_lbl = f"{NOTE[f['root']]}{f['q']}" if f else None
    bars_out.append({
        "i": i,
        "draft": d_lbl if d_lbl is not None else (last_draft or ""),
        "final": f_lbl if f_lbl is not None else (last_final or ""),
        "root_pc": f["root"] if f else (draft_bars[i - 1]["root"] if i > 0 else 0),
        "t0": round((f or d or {}).get("t0", 0.0), 2),
        "held": f is None,
    })
    if d_lbl:
        last_draft = d_lbl
    if f_lbl:
        last_final = f_lbl

out = {
    "title": TITLE,
    "tempo_bpm": round(final.tempo_bpm, 1),
    "key": final.global_key,
    "duration_s": round(final.duration_s, 1),
    "n_bars": n_bars,
    "bars": bars_out,
    "timing": {"draft_s": round(t_draft, 1), "final_s": round(t_final, 1)},
}
Path("scratchpad/demo_chord_data.json").write_text(json.dumps(out, indent=1))
n_changed = sum(1 for b in bars_out if b["draft"] != b["final"])
print(f"bars where draft != final: {n_changed}/{n_bars}")
