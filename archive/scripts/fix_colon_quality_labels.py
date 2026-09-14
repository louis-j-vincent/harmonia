"""One-off retroactive fix for the colon-label bug (docs/known_issues.md).

chord_pipeline_v1.py always emits labels as "D:maj7" (colon-separated), but
render_youtube_chart.py's _split_label/_QUALITY_TO_IREAL tables were written
for a concatenated format and a different quality-name vocabulary — every
quality token baked from this path is either the raw corrupted ":maj7" string
or (for hdim7/dim7/minmaj7) an unmapped raw sev_h name, at all three
family/seventh/exact levels identically (the level-collapse step silently
no-op'd on the unrecognised string, so all three levels show the same raw
value instead of the correctly-collapsed one).

Both bugs are now fixed at the source (scripts/render_youtube_chart.py). This
script repairs already-baked chart JSON without re-running inference: root,
bass and confidence were never touched by the bug (parse_token's root/bass
extraction succeeded despite the colon), so only the "q" string at each level
needs recomputing from the recovered raw sev_h.

Does NOT touch: segmentation, confidence values, suggestion data (the
suggestion feature postdates these files and doesn't exist in them yet).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
PLOTS_DIR = REPO / "docs" / "plots"

from render_youtube_chart import _QUALITY_TO_FAMILY, _QUALITY_TO_IREAL  # noqa: E402


def _recover_sev_h(corrupted_q: str) -> str:
    """A corrupted level's "q" string is either ":sev_h" (colon bug) or a raw
    sev_h name that fell through the old vocabulary-mismatch (hdim7/dim7/
    minmaj7). Strip a leading colon if present; otherwise it's already the
    raw name."""
    return corrupted_q[1:] if corrupted_q.startswith(":") else corrupted_q


def _is_corrupted(q: str) -> bool:
    if q.startswith(":"):
        return True
    return q in ("hdim7", "dim7", "minmaj7")


def fix_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"const P = (\{.*?\});\n", text)
    if not m:
        return 0
    P = json.loads(m.group(1))
    n_fixed = 0
    for c in P.get("chords", []):
        lv = c.get("lv", {})
        # All three levels show the same raw corrupted string when broken
        # (see module docstring) — recover once from whichever is present.
        raw = None
        for level in ("exact", "seventh", "family"):
            q = lv.get(level, {}).get("q", "")
            if _is_corrupted(q):
                raw = _recover_sev_h(q)
                break
        if raw is None:
            continue
        exact_ireal = _QUALITY_TO_IREAL.get(raw, raw)
        family_name = _QUALITY_TO_FAMILY.get(raw, raw)
        family_ireal = _QUALITY_TO_IREAL.get(family_name, family_name)
        lv["exact"]["q"] = exact_ireal
        lv["seventh"]["q"] = exact_ireal  # this pipeline never emits >7th extensions
        lv["family"]["q"] = family_ireal
        n_fixed += 1

    if n_fixed:
        new_json = json.dumps(P)
        text = text[: m.start(1)] + new_json + text[m.end(1) :]
        path.write_text(text, encoding="utf-8")
    return n_fixed


def main() -> None:
    total = 0
    for f in sorted(PLOTS_DIR.glob("inferred_*.html")):
        n = fix_file(f)
        if n:
            print(f"fixed {n} chords in {f.name}")
            total += n
    print(f"\n{total} chords repaired")


if __name__ == "__main__":
    main()
