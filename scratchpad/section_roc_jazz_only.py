"""section_roc_jazz_only.py — 2026-07-18, section-level suggestion tool,
follow-up diagnostic run triggered by a genuinely surprising negative result
in section_roc_suggest.py's full-corpus run: pooled ROC-AUC was only 0.611
(grain=8) / 0.567 (grain=4) — MUCH weaker than the bar-level chord-identity
task's 0.99. Hypothesis (root-caused via a quick jazz-only vs full-corpus
comparison, see docs/known_issues.md entry): the FULL iReal corpus spans 7
playlists (jazz1460, pop400, blues50, brazilian220, country, dixieland1,
latin_salsa50), and many non-jazz genres in that mix are vamp/loop-based
(the SAME 2-4-chord progression recycled across verse/chorus/bridge), so
"different named section" negatives in those genres are often harmonically
near-identical to "same section" positives — a real property of the genre,
not a bug in the pair-pool construction. Confirmed: restricting to
jazz1460-only raised grain=8 ROC-AUC 0.611->0.696 (quick uncalibrated
check). This script re-runs the FULL nested ROC/recall-target methodology
(section_roc_suggest.py, unchanged) on the jazz1460-only subset, since the
user's own worked example (Autumn Leaves) and the 3 real-audio validation
songs are all jazz/pop-standard repertoire, not vamp-based pop/latin/country
— the jazz-only numbers are the honestly relevant calibration target for
this tool, not the diluted full-corpus number.
"""
from __future__ import annotations
import sys, json, io, time
from pathlib import Path
from contextlib import redirect_stdout
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from tau_auto_search import load_corpus_bar_chords, split_songs_3way
from harmonia.data.ireal_corpus import load_playlist
from section_pairs import build_section_pairs
from section_roc_suggest import run_grain, RECALL_TARGETS, SEEDS

OUT_DIR = Path(__file__).resolve().parent


def main():
    t0 = time.time()
    corpus = load_corpus_bar_chords(max_tunes=None)
    buf = io.StringIO()
    with redirect_stdout(buf):
        jazz_tunes = load_playlist(Path("data/ireal/jazz1460.txt"))
    jazz_titles = set(t.title for t in jazz_tunes)
    jazz_corpus = [c for c in corpus if c["title"] in jazz_titles]
    print("jazz1460-matched tunes in corpus: %d / %d total" % (len(jazz_corpus), len(corpus)))

    out = {"recall_targets": RECALL_TARGETS, "seeds": SEEDS, "n_jazz_tunes": len(jazz_corpus), "grains": {}}
    for grain in (8, 4):
        out["grains"][str(grain)] = run_grain(jazz_corpus, grain)

    (OUT_DIR / "section_roc_jazz_only_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote section_roc_jazz_only_results.json, elapsed %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
