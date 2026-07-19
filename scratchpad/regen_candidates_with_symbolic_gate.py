"""regen_candidates_with_symbolic_gate.py — 2026-07-18, task 3 of the
dual-matrix cross-validation continuation call.

Regenerates the 3 real songs' candidate JSON (both the UI-facing top-20
`bar_merge_candidates_<slug>.json` and the uncapped
`bar_merge_full_census_<slug>.json`) using the NEW joint audio+symbolic
auto-tier gate (`bar_merge_candidates.apply_symbolic_gate`, TAU_SYMBOLIC=
0.90) on top of the unchanged `candidate_groups()` generator. Reuses the
already-cached `baseline_chords_<slug>.json` decodes from tonight (no
re-decode). See `bar_merge_candidates.py`'s module-level TAU_SYMBOLIC
docstring and `docs/known_issues.md` for the full derivation + the
explicit "do not ship auto-apply" caveat -- this script changes which
candidates are LABELED tier=="auto" vs "suggest" in the served static
JSON; it does NOT apply any merge and does NOT touch any server/UI code.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bar_merge_candidates import candidate_groups, apply_symbolic_gate
from auto_apply_merges import SONGS, AUDIO_DIR
from realaudio_threshold_check import get_baseline_chords

OUT_DIR = Path(__file__).resolve().parent


def main():
    summary = {}
    for slug, cfg in SONGS.items():
        audio_path = AUDIO_DIR / cfg["audio_name"]
        base_chords = get_baseline_chords(slug)  # cached, no re-decode

        # UI-facing, capped at 20
        cand_ui, meta_ui = candidate_groups(audio_path, max_candidates=20)
        n_auto_before = sum(1 for c in cand_ui if c["tier"] == "auto")
        cand_ui_gated = apply_symbolic_gate(cand_ui, base_chords)
        n_auto_after = sum(1 for c in cand_ui_gated if c["tier"] == "auto")
        meta_ui["n_auto"] = n_auto_after
        meta_ui["n_suggest"] = sum(1 for c in cand_ui_gated if c["tier"] == "suggest")
        meta_ui["symbolic_gate_applied"] = True
        meta_ui["tau_symbolic"] = 0.90
        (OUT_DIR / f"bar_merge_candidates_inferred_{slug}.json").write_text(
            json.dumps({"candidates": cand_ui_gated, "meta": meta_ui}, indent=2))

        # full uncapped census
        cand_full, meta_full = candidate_groups(audio_path, max_candidates=100000)
        n_auto_full_before = sum(1 for c in cand_full if c["tier"] == "auto")
        cand_full_gated = apply_symbolic_gate(cand_full, base_chords)
        n_auto_full_after = sum(1 for c in cand_full_gated if c["tier"] == "auto")
        meta_full["n_auto"] = n_auto_full_after
        meta_full["n_suggest"] = sum(1 for c in cand_full_gated if c["tier"] == "suggest")
        meta_full["symbolic_gate_applied"] = True
        meta_full["tau_symbolic"] = 0.90
        (OUT_DIR / f"bar_merge_full_census_{slug}.json").write_text(
            json.dumps({"candidates": cand_full_gated, "meta": meta_full}, indent=2))
        assert (OUT_DIR / f"bar_merge_full_census_{slug}.json").exists()

        summary[slug] = {
            "ui_capped_n_auto_before": n_auto_before, "ui_capped_n_auto_after": n_auto_after,
            "full_census_n_auto_before": n_auto_full_before, "full_census_n_auto_after": n_auto_full_after,
            "full_census_n_total": len(cand_full_gated),
        }
        print(f"{slug}: UI-capped auto {n_auto_before}->{n_auto_after}; "
              f"full census auto {n_auto_full_before}->{n_auto_full_after} "
              f"(of {len(cand_full_gated)} total candidates)", file=sys.stderr)

    (OUT_DIR / "regen_candidates_with_symbolic_gate_summary.json").write_text(
        json.dumps(summary, indent=2))
    print("\nwrote regen_candidates_with_symbolic_gate_summary.json", file=sys.stderr)


if __name__ == "__main__":
    main()
