"""Re-profile infer_chords_v1(nnls24) stage-by-stage after the 2026-07-19
pipeline additions (bestfit beat period, flux/structure grid anchor,
barlocked sections, N-chord gate, Occam post-pass, calibration).

Matches the LIVE server defaults exactly (scripts/harmonia_server.py
_ANALYZE_* + beat_period_mode):
  feature_frontend=nnls24, bass_frontend=musx, quality_frontend=musx,
  segment_source=nnls, beat_period_mode=bestfit,
  HARMONIA_SECTION_MODE=barlocked (default), HARMONIA_GRID_ANCHOR=flux (default),
  HARMONIA_OCCAM_POSTPASS=1 (default), HARMONIA_NNLS24_CALIB=on (default).

Usage: run twice on the SAME audio path — first call is COLD (nnls/musx
caches empty for this stem), second is WARM (stem-keyed cache hit) — to
separate one-time cost from steady-state cost, same methodology as the
2026-07-17 profiling doc.
"""
import cProfile
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from harmonia.models.chord_pipeline_v1 import infer_chords_v1

AUDIO = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/henny_prof.wav")
LABEL = sys.argv[2] if len(sys.argv) > 2 else "run"

pr = cProfile.Profile()
t0 = time.perf_counter()
pr.enable()
chart = infer_chords_v1(
    AUDIO,
    feature_frontend="nnls24",
    bass_frontend="musx",
    quality_frontend="musx",
    segment_source="nnls",
    beat_period_mode="bestfit",
    cache_dir=Path("data/cache"),
)
pr.disable()
t1 = time.perf_counter()

print(f"[{LABEL}] TOTAL wall time: {t1-t0:.2f}s  "
      f"chords={len(chart.chords)} sections={len(chart.sections)} "
      f"key={chart.global_key} tempo={chart.tempo_bpm}")

stats = pstats.Stats(pr)
stats.sort_stats("cumulative")

# Only show frames from our own codebase (not every numpy/torch internal) plus
# a few named library entry points we care about (subprocess, vamp, librosa
# beat_track) so the table stays readable.
KEEP = ("chord_pipeline_v1.py", "nnls_features.py", "musx_bass.py",
        "section_structure.py", "beat.py", "subprocess.py", "vamp",
        "soundfile.py")
stats.print_stats(0)  # header only (suppresses noise before filtering below)

import io
buf = io.StringIO()
s2 = pstats.Stats(pr, stream=buf)
s2.sort_stats("cumulative")
s2.print_stats(60)
lines = buf.getvalue().splitlines()
print(f"\n--- filtered cProfile (cumulative time, {LABEL}) ---")
for ln in lines:
    if any(k in ln for k in KEEP) or "cumulative" in ln:
        print(ln)

pr.dump_stats(f"scratchpad/profile_{LABEL}.pstats")
