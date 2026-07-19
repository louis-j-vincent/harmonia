"""build_k_prior.py — extract (n_bars, n_distinct_sections) from the FULL
iReal corpus (all 7 playlist files, ~2401 tunes, not a sample), to build an
empirical P(k | n_bars) prior for principled k-selection in the real-audio
section-clustering pipeline.

k here = number of DISTINCT section labels (A/B/C/...) in a tune, counted
via `sectionized_measures()` (real per-bar section labels from iReal's own
*A/*B/... markers, survives repeat-expansion). n_bars = number of measures
in the flattened chart (post repeat-expansion, i.e. the same "as-performed"
bar count convention this project uses for real-audio n_blocks*8).

Only tunes with >=2 distinct sections are "multi-section" (matches the
brief's ~1992-tune framing approximately, single-section/blues-form-only
tunes are excluded from the correlation fit but their count is reported).
"""
from __future__ import annotations
import io
import json
import sys
from collections import defaultdict, Counter
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, "/Users/vincente/Documents/Projets Perso/Code/harmonia")
from harmonia.data.ireal_corpus import load_playlist, sectionized_measures

REPO = Path(__file__).resolve().parents[3]  # not reliable, override below
DATA_DIR = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia/data/ireal")
OUT_PATH = Path(__file__).resolve().parent / "k_prior_corpus_extract.json"


def extract_all():
    files = sorted(DATA_DIR.glob("*.txt"))
    records = []
    fail_count = 0
    for f in files:
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                tunes = load_playlist(f)
            except Exception:
                continue
        for t in tunes:
            try:
                buf2 = io.StringIO()
                with redirect_stdout(buf2):
                    sm = sectionized_measures(t)
            except Exception:
                fail_count += 1
                continue
            if not sm:
                fail_count += 1
                continue
            n_bars = len(sm)
            labels_in_order = [lab for lab, _ in sm]
            distinct = sorted(set(labels_in_order))
            k = len(distinct)
            records.append({
                "title": t.title,
                "source_file": f.name,
                "style": getattr(t, "style", None),
                "n_bars": n_bars,
                "k": k,
                "distinct_labels": distinct,
            })
    return records, fail_count


def main():
    records, fail_count = extract_all()
    print(f"Extracted {len(records)} tunes ({fail_count} failed to parse/flatten)")
    multi = [r for r in records if r["k"] >= 2]
    single = [r for r in records if r["k"] == 1]
    print(f"Multi-section (k>=2): {len(multi)}; single-section (k==1): {len(single)}")

    OUT_PATH.write_text(json.dumps({
        "n_total_parsed": len(records),
        "n_failed": fail_count,
        "n_multi_section": len(multi),
        "n_single_section": len(single),
        "records": records,
    }, indent=1))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
