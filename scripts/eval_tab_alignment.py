"""
eval_tab_alignment.py — benchmark guitar tab alignment against iReal pop corpus.

For each song in pop400.txt that has a well-rated UG guitar tab:
  1. Parse the iReal ground-truth chord sequence
  2. Fetch the best UG tab
  3. Parse the tab chord tokens
  4. Align with DTW (try all 12 transpositions)
  5. Score: exact / family / mismatch / gap rates

Run:
    .venv/bin/python scripts/eval_tab_alignment.py [--songs N] [--out results.json]

Results are saved to JSON so you can re-run analysis without re-fetching tabs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.data.ireal_corpus import load_playlist, tune_to_mma
from harmonia.tab_aligner import _parse_ireal, align_tab_to_chart
from harmonia.tab_renderer import parse_tab


# ── helpers ────────────────────────────────────────────────────────────────────

_ARTICLES = re.compile(r"^(the|a|an)\s+|\s+(the|a|an)$")
_NONALNUM  = re.compile(r"[^a-z0-9 ]+")


def _canon_title(s: str) -> str:
    """Canonical form for exact title comparison.

    Handles iReal-specific quirks before normalising:
      • "(Page 1)" / " 1" / " 2" suffixes on multi-page charts
      • "(Artist Name)" disambiguation appended to title
      • "(Naturally)" style parenthetical kept (it's part of the title)
      • ", Babe" / ", Baby" subtitle suffixes
    Then:
      1. Lowercase
      2. Drop apostrophes/hyphens in-place (don't→dont)
      3. Replace remaining non-alphanumeric with space
      4. Collapse whitespace
      5. Strip leading/trailing articles ("The", "A", "An")

    Examples:
      "Ain't No Sunshine"         →  "aint no sunshine"
      "Circle Game, The"          →  "circle game"
      "Dancing In The Dark 1"     →  "dancing in the dark"
      "Deacon Blues (Page 1)"     →  "deacon blues"
      "Crazy (Gnarls Barkley)"    →  "crazy"
      "Don't Stop Me Now"         →  "dont stop me now"
    """
    # Strip iReal page markers: "(Page N)" or trailing " N" (single digit)
    s = re.sub(r"\s*\(page\s*\d+\)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+\d+\s*$", "", s).strip()
    # Strip artist-as-disambiguator in parens: "Crazy (Gnarls Barkley)"
    # Heuristic: if the parenthetical contains 2+ words or looks like a band name,
    # drop it.  Keep short single-word ones like "(Naturally)" that are part of titles.
    def _strip_artist_paren(m):
        inner = m.group(1).strip()
        # Keep if it's clearly part of the title (≤1 word, or starts with a verb/adj)
        if len(inner.split()) <= 1:
            return f" {inner} "   # keep
        return " "                # drop — looks like an artist name
    s = re.sub(r"\(([^)]+)\)", _strip_artist_paren, s)
    # Collapse any remaining artifacts
    s = s.strip().rstrip(",").strip()

    # Standard normalisation
    s = s.lower()
    s = re.sub(r"[''`\-]", "", s)          # apostrophes, hyphens in-place
    s = _NONALNUM.sub(" ", s)              # other punct → space
    s = re.sub(r"\s+", " ", s).strip()
    s = _ARTICLES.sub("", s).strip()       # leading/trailing article
    return s


def _title_match(query: str, candidate: str) -> bool:
    """Exact match after canonical normalization."""
    return _canon_title(query) == _canon_title(candidate)


def _artist_words(s: str) -> frozenset[str]:
    """Canonical word set for an artist name.

    Strips parentheticals, lowercases, strips non-alphanumeric, removes
    single-char tokens and articles.  Order-independent so iReal's inverted
    names ("Withers Bill") match UG's normal order ("Bill Withers").
    """
    s = re.sub(r"\(.*?\)", "", s)          # "(Lennon, McCartney)" → ""
    s = re.sub(r"[''`]", "", s.lower())   # apostrophes in-place; lowercase
    s = _NONALNUM.sub(" ", s)             # hyphens and other punct → space
    words = {w for w in s.split() if len(w) >= 2}
    words -= {"the", "and", "feat", "ft", "vs", "von", "de", "la", "le"}
    return frozenset(words)


def _artist_match(ireal_composer: str, ug_artist: str) -> bool:
    """Exact word-set match after canonical normalisation.

    Handles iReal name inversions ("Withers Bill" ↔ "Bill Withers").
    For hyphenated multi-credits ("Elton John-Tim Rice", "Marvin Gaye-Tammi Terrell"),
    each hyphen-segment is tried independently — UG lists the performer, not the full
    songwriter credit.
    """
    cw = _artist_words(ug_artist)
    if not cw:
        return False
    # Strip parenthetical lyricist credit first, then try each hyphen-segment
    base = re.sub(r"\s*\(.*?\)", "", ireal_composer).strip()
    segments = base.split("-") if "-" in base else [base]
    return any(_artist_words(seg) == cw for seg in segments)


def ireal_timeline_to_payload(chart) -> list[dict]:
    """Convert MMAChart.timeline to the chart_chords_payload format expected
    by align_tab_to_chart: list of {"root": pc, "lv": {"seventh": {"q": quality}}}."""
    payload = []
    for _barno, _label, slots in chart.timeline:
        for _beat, ireal_tok, _mma in slots:
            pc, q = _parse_ireal(ireal_tok)
            payload.append({"root": pc, "lv": {"seventh": {"q": q}}})
    return payload


def tab_to_chord_tokens(raw_content: str, bpb: int = 4) -> list[str]:
    """Extract flat ordered chord token list from a UG tab string."""
    bars = parse_tab(raw_content, bpb=bpb)
    tokens: list[str] = []
    for bar in bars:
        for chord in bar["chords"]:
            tokens.append(chord["ireal"])
    return tokens


def score_alignment(annotations) -> dict:
    total = len(annotations)
    if total == 0:
        return {"total": 0, "exact": 0, "family": 0, "mismatch": 0, "gap": 0,
                "exact_pct": 0, "family_pct": 0, "mismatch_pct": 0, "gap_pct": 0,
                "mean_dist": 1.0}
    counts = {"exact": 0, "family": 0, "mismatch": 0, "gap": 0}
    for ann in annotations:
        counts[ann.match] = counts.get(ann.match, 0) + 1
    mean_dist = round(sum(a.dist for a in annotations) / total, 3)
    return {
        "total":        total,
        **counts,
        "exact_pct":    round(100 * counts["exact"]    / total, 1),
        "family_pct":   round(100 * counts["family"]   / total, 1),
        "mismatch_pct": round(100 * counts["mismatch"] / total, 1),
        "gap_pct":      round(100 * counts["gap"]      / total, 1),
        "mean_dist":    mean_dist,   # pitch-class Jaccard distance ∈ [0,1], lower = better
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs",  type=int, default=80,
                    help="Max songs to evaluate (default 80)")
    ap.add_argument("--min-chords", type=int, default=20,
                    help="Skip songs with fewer GT chords (default 20)")
    ap.add_argument("--min-tab-rating", type=float, default=3.5,
                    help="Minimum UG tab star rating to accept (default 3.5)")
    ap.add_argument("--delay",  type=float, default=1.5,
                    help="Seconds between UG requests (default 1.5)")
    ap.add_argument("--out",    type=Path,
                    default=REPO / "docs" / "tab_alignment_eval.json",
                    help="Output JSON path")
    ap.add_argument("--resume", action="store_true",
                    help="Skip songs already in the output file")
    args = ap.parse_args()

    from harmonia.tab_fetcher import search_tabs, fetch_tab_chords

    # Load existing results if resuming
    existing: dict[str, dict] = {}
    if args.resume and args.out.exists():
        existing = {r["title"]: r for r in json.loads(args.out.read_text())}
        print(f"Resuming: {len(existing)} songs already done.")

    tunes = load_playlist(REPO / "data" / "ireal" / "pop400.txt")
    # Filter 4/4 songs only (tabs are almost exclusively 4/4)
    tunes = [t for t in tunes if t.time_signature and t.time_signature[0] == 4]
    print(f"4/4 pop songs: {len(tunes)}")

    results: list[dict] = []
    n_done = n_found = n_aligned = 0

    for tune in tunes:
        if n_done >= args.songs:
            break

        title    = tune.title
        composer = tune.composer or ""

        # Parse composer to get artist name (composer field is "Composer (Lyricist)" style)
        artist_raw = re.sub(r"\s*\(.*?\)", "", composer).strip()

        # Skip if already evaluated
        if args.resume and title in existing:
            results.append(existing[title])
            n_done += 1
            continue

        # Build ground-truth chord payload
        try:
            chart = tune_to_mma(tune)
        except Exception as e:
            print(f"  SKIP {title!r}: MMA parse error: {e}")
            continue

        gt_payload = ireal_timeline_to_payload(chart)
        if len(gt_payload) < args.min_chords:
            print(f"  SKIP {title!r}: only {len(gt_payload)} GT chords")
            continue

        n_done += 1
        print(f"\n[{n_done:3d}] {title!r}  ({artist_raw})  [{len(gt_payload)} GT chords]")

        # Fetch best UG tab
        try:
            tab_results = search_tabs(title, artist_raw)
            time.sleep(args.delay)
        except Exception as e:
            print(f"       search failed: {e}")
            results.append({"title": title, "composer": composer, "status": "search_error",
                             "error": str(e)})
            continue

        if not tab_results:
            print(f"       no tabs found")
            results.append({"title": title, "composer": composer, "status": "no_tab"})
            continue

        # Require exact canonical title match + fuzzy artist match.
        # Exact title: "Caught Up In You" ≠ "Caught Up In The Rapture".
        # Fuzzy artist: handles iReal name inversions ("Withers Bill" ↔ "Bill Withers").
        title_matched = [r for r in tab_results
                         if _title_match(title, r.song_name)
                         and _artist_match(composer, r.artist_name)]

        if not title_matched:
            # Debug: show why each candidate failed
            def _why(r):
                t = "T✓" if _title_match(title, r.song_name) else f"T✗({_canon_title(r.song_name)!r})"
                a = "A✓" if _artist_match(composer, r.artist_name) else f"A✗({r.artist_name!r})"
                return f"{r.song_name!r} {t} {a}"
            scores_str = " | ".join(_why(r) for r in tab_results[:3])
            print(f"       no match — {scores_str}")
            results.append({"title": title, "composer": composer, "status": "title_mismatch",
                             "candidates": [f"{r.song_name}/{r.artist_name}" for r in tab_results[:3]]})
            continue

        # Pick best by rating × votes among acceptable ratings
        good = [r for r in title_matched
                if r.rating >= args.min_tab_rating and r.tab_type in ("Chords", "Tab")]
        if not good:
            good = title_matched
        best = max(good, key=lambda r: r.rating * (1 + 0.1 * min(r.votes, 500)))
        n_found += 1
        print(f"       tab: {best.rating:.2f}★ ({best.votes}v) — {best.tab_type} — {best.song_name}")

        # Fetch raw content
        try:
            tab = fetch_tab_chords(best)
            time.sleep(args.delay)
        except Exception as e:
            print(f"       fetch failed: {e}")
            results.append({"title": title, "composer": composer, "status": "fetch_error",
                             "error": str(e), "tab_rating": best.rating, "tab_votes": best.votes})
            continue

        if tab is None:
            print(f"       fetch returned None")
            results.append({"title": title, "composer": composer, "status": "fetch_none",
                             "tab_rating": best.rating, "tab_votes": best.votes})
            continue

        # Parse tab → chord token list
        tab_tokens = tab_to_chord_tokens(tab.raw_content, bpb=chart.beats_per_bar)
        if not tab_tokens:
            print(f"       no chords parsed from tab")
            results.append({"title": title, "composer": composer, "status": "empty_tab",
                             "tab_rating": best.rating})
            continue

        print(f"       tab chords: {len(tab_tokens)}")

        # Align
        try:
            result = align_tab_to_chart(
                gt_payload, tab_tokens,
                tab_rating=best.rating, tab_votes=best.votes,
            )
        except Exception as e:
            print(f"       alignment error: {e}")
            results.append({"title": title, "composer": composer, "status": "align_error",
                             "error": str(e)})
            continue

        scores = score_alignment(result.annotations)
        n_aligned += 1
        print(f"       dtw={result.dtw_cost:.3f}  exact={scores['exact_pct']}%  "
              f"family={scores['family_pct']}%  mismatch={scores['mismatch_pct']}%  "
              f"gap={scores['gap_pct']}%  dist={scores['mean_dist']:.3f}  "
              f"transpose={result.transpose_semitones}st")

        results.append({
            "title":              title,
            "composer":           composer,
            "status":             "ok",
            "gt_chords":          len(gt_payload),
            "tab_chords":         len(tab_tokens),
            "tab_rating":         round(best.rating, 3),
            "tab_votes":          best.votes,
            "tab_type":           best.tab_type,
            "tab_song_name":      best.song_name,
            "dtw_cost":           result.dtw_cost,
            "transpose_semitones": result.transpose_semitones,
            **scores,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    ok   = [r for r in results if r.get("status") == "ok"]
    print(f"\n{'='*60}")
    print(f"Songs evaluated:  {n_done}")
    print(f"Tabs found:       {n_found}")
    print(f"Aligned:          {n_aligned}")
    if ok:
        avg = lambda key: round(sum(r[key] for r in ok) / len(ok), 1)
        print(f"\nMean over {len(ok)} aligned songs:")
        print(f"  DTW cost:    {avg('dtw_cost')}")
        print(f"  Exact:       {avg('exact_pct')}%")
        print(f"  Family:      {avg('family_pct')}%")
        print(f"  Mismatch:    {avg('mismatch_pct')}%")
        print(f"  Gap:         {avg('gap_pct')}%")

        # Sorted by exact% to see best/worst
        by_exact = sorted(ok, key=lambda r: -r["exact_pct"])
        print(f"\nTop 5 (exact%):")
        for r in by_exact[:5]:
            print(f"  {r['title']!r:40s} exact={r['exact_pct']}%  dtw={r['dtw_cost']:.3f}")
        print(f"Bottom 5 (exact%):")
        for r in by_exact[-5:]:
            print(f"  {r['title']!r:40s} exact={r['exact_pct']}%  dtw={r['dtw_cost']:.3f}")

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {args.out}")


if __name__ == "__main__":
    main()
