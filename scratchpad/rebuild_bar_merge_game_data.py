"""rebuild_bar_merge_game_data.py — regenerate per-song candidate JSON +
the combined bar_merge_game_data.json consumed by /debug/bar-merge-game,
using bar_merge_candidates.candidate_groups()'s current default algorithm
(k-NN top-1 edge selection, no transitive closure — 2026-07-18 bakeoff
winner, see bar_merge_candidates.py module docstring for the full
rationale and known_issues.md's ★ CHORD-ROBUSTNESS / BAR-MERGE entry).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bar_merge_candidates import candidate_groups

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent

SONGS = {
    "inferred_aretha_franklin_chain_of_fools_official_lyric_video.html": {
        "title": "Aretha Franklin — Chain of Fools",
        "audio_name": "aretha_franklin_chain_of_fools_official_lyric_video.m4a",
    },
    "inferred_autumn_leaves.html": {
        "title": "Autumn Leaves",
        "audio_name": "autumn_leaves.m4a",
    },
    "inferred_abba_chiquitita_official_lyric_video.html": {
        "title": "ABBA — Chiquitita",
        "audio_name": "abba_chiquitita_official_lyric_video.m4a",
    },
}


def main():
    combined = {}
    for chart_file, meta in SONGS.items():
        audio_path = REPO / "docs" / "audio" / meta["audio_name"]
        print("=== %s ===" % chart_file)
        candidates, cmeta = candidate_groups(audio_path)  # defaults: algo="knn", k=1, tau=0.93
        print("  %d candidates, n_bars_total=%d, algo=%s k=%s" %
              (len(candidates), cmeta["n_bars_total"], cmeta["algo"], cmeta["k"]))
        per_song = {"chart_file": chart_file, "audio_name": meta["audio_name"],
                    "meta": cmeta, "candidates": candidates}
        slug = chart_file.replace(".html", "").replace("inferred_", "")
        (OUT_DIR / ("bar_merge_candidates_inferred_%s.json" % slug)).write_text(
            json.dumps(per_song, indent=2))
        combined[chart_file] = {"title": meta["title"], "meta": cmeta, "candidates": candidates}

    (OUT_DIR / "bar_merge_game_data.json").write_text(json.dumps(combined, indent=2))
    print("\nwrote bar_merge_game_data.json (%d songs)" % len(combined))


if __name__ == "__main__":
    main()
