"""
eval_tab_alignment_audio.py — audio-grounded guitar tab alignment eval.

Pipeline per song:
  1. Search YouTube → download audio (cached as WAV)
  2. Run Harmonia chord pipeline on the audio (cached as JSON)
  3. Fetch best UG guitar tab (same title+artist filter as eval_tab_alignment.py)
  4. DTW-align tab chord sequence against the audio chord sequence
     → each tab chord gets a timestamp and bar/beat from the audio grid
  5. Score:
     - tab↔audio dist  : Jaccard distance between tab chord and what Harmonia hears
     - audio↔ireal dist: Jaccard distance between audio chord and iReal GT
       (baseline: how well the pipeline itself does on this audio)

Run:
    .venv/bin/python scripts/eval_tab_alignment_audio.py [--songs N] [--out results.json]

Cache dirs (created automatically):
    data/cache/yt_audio/   — downloaded WAVs
    data/cache/yt_charts/  — Harmonia JSON outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.data.ireal_corpus import load_playlist, tune_to_mma
from harmonia.tab_aligner import (
    _parse_ireal, _chord_dist, _chord_tones,
    align_tab_to_chart,
    audio_chart_to_sequence, align_tab_to_audio,
)
from harmonia.tab_renderer import parse_tab

logging.basicConfig(level=logging.WARNING)   # suppress pipeline INFO spam

# ── reuse matching helpers from the existing eval ─────────────────────────────
_ARTICLES = re.compile(r"^(the|a|an)\s+|\s+(the|a|an)$")
_NONALNUM  = re.compile(r"[^a-z0-9 ]+")


def _canon_title(s: str) -> str:
    s = re.sub(r"\s*\(page\s*\d+\)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+\d+\s*$", "", s).strip()
    def _strip_artist_paren(m):
        inner = m.group(1).strip()
        return f" {inner} " if len(inner.split()) <= 1 else " "
    s = re.sub(r"\(([^)]+)\)", _strip_artist_paren, s)
    s = s.strip().rstrip(",").strip()
    s = s.lower()
    s = re.sub(r"[''`\-]", "", s)
    s = _NONALNUM.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _ARTICLES.sub("", s).strip()
    return s


def _artist_words(s: str) -> frozenset[str]:
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[''`]", "", s.lower())
    s = _NONALNUM.sub(" ", s)
    words = {w for w in s.split() if len(w) >= 2}
    words -= {"the", "and", "feat", "ft", "vs", "von", "de", "la", "le"}
    return frozenset(words)


def _artist_match(ireal_composer: str, ug_artist: str) -> bool:
    cw = _artist_words(ug_artist)
    if not cw:
        return False
    base = re.sub(r"\s*\(.*?\)", "", ireal_composer).strip()
    segments = base.split("-") if "-" in base else [base]
    return any(_artist_words(seg) == cw for seg in segments)


def _title_match(q: str, c: str) -> bool:
    return _canon_title(q) == _canon_title(c)


# ── YouTube download ──────────────────────────────────────────────────────────

def _slug(s: str) -> str:
    """Safe filename slug."""
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s[:80]


def download_audio(title: str, artist: str, cache_dir: Path) -> Path | None:
    """Search YouTube and download best audio as WAV. Returns path or None."""
    import yt_dlp

    slug = _slug(f"{title}_{artist}")
    wav_path = cache_dir / f"{slug}.mp3"
    if wav_path.exists():
        return wav_path

    query = f"ytsearch1:{title} {artist} audio"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": "bestaudio/best",
        "outtmpl": str(cache_dir / f"{slug}.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",   # smaller than wav; librosa + Basic Pitch handle mp3
            "preferredquality": "192",
        }],
        # limit duration: skip anything over 10 min (likely a live set)
        "match_filter": yt_dlp.utils.match_filter_func("duration < 600"),
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
            if not info or not info.get("entries"):
                return None
            entry = info["entries"][0]
            print(f"       yt: {entry['title']!r} ({entry['duration']}s)")
            ydl.download([entry["webpage_url"]])
        return wav_path if wav_path.exists() else None
    except Exception as e:
        print(f"       yt download failed: {e}")
        return None


# ── Harmonia pipeline (cached) ────────────────────────────────────────────────

def run_harmonia(audio_path: Path, chart_dir: Path, delete_audio_after: bool = True) -> dict | None:
    """Run HarmoniaPipeline (YouTube-tuned) and cache result as JSON.

    Parameters tuned for YouTube pop audio (v3):
      self_transition_boost  1.0  — compromise: less over-seg than 0.5, more than 2.0
      compress_emission      sqrt — modest known gain on noisy audio
      chroma_change_scale    1.5  — HCDF/TCS boundary signal (replaces cosine);
                                    scale 1.5 = full mobility at TCS dist ≥ 1.33
      key_weight_scale       1.0  — downweights chromatic passing tones
      min_segment_beats      4    — finer structural cuts
      tempo_adaptive min_chord_beats handled automatically in pipeline.py
    """
    chart_path = chart_dir / (audio_path.stem + ".json")
    if chart_path.exists():
        return json.loads(chart_path.read_text())

    from harmonia.pipeline import HarmoniaPipeline
    pipeline = HarmoniaPipeline(
        self_transition_boost=1.0,       # up from 0.5 — reduces over-segmentation
        compress_emission="sqrt",        # mild dynamic range compression
        # emission_scoring="dot" kept as default — cosine breaks with STB<1
        chroma_change_scale=1.5,         # HCDF/TCS signal; was 2.0 cosine
        key_weight_scale=1.0,            # downweight chromatic passing tones
        min_segment_beats=4,             # finer structural cuts (was 8)
    )
    try:
        chart = pipeline.run(audio_path)
        chart.save_json(chart_path)
        if delete_audio_after:
            audio_path.unlink(missing_ok=True)
        return json.loads(chart_path.read_text())
    except Exception as e:
        print(f"       harmonia failed: {e}")
        return None


# ── iReal GT helpers ──────────────────────────────────────────────────────────

def ireal_to_flat_seq(chart) -> list[tuple[int, str]]:
    """Flatten MMAChart.timeline to [(pc, quality)] in bar-beat order."""
    seq = []
    for _barno, _label, slots in chart.timeline:
        for _beat, ireal_tok, _mma in slots:
            pc, q = _parse_ireal(ireal_tok)
            seq.append((pc, q))
    return seq


def ireal_bar_beat_lookup(chart) -> dict[tuple[int, int], tuple[int, str]]:
    """Build {(bar, beat): (pc, quality)} from MMAChart.timeline."""
    lookup = {}
    for barno, _label, slots in chart.timeline:
        for beat, ireal_tok, _mma in slots:
            pc, q = _parse_ireal(ireal_tok)
            lookup[(barno, beat)] = (pc, q)
    return lookup


# ── tab helpers ───────────────────────────────────────────────────────────────

def tab_to_chord_tokens(raw_content: str, bpb: int = 4) -> list[str]:
    bars = parse_tab(raw_content, bpb=bpb)
    return [c["ireal"] for bar in bars for c in bar["chords"]]


# ── scoring ───────────────────────────────────────────────────────────────────

def score_placements(placements) -> dict:
    """Aggregate tab↔audio distances from AudioAlignmentResult.placements."""
    if not placements:
        return {"n": 0, "mean_tab_audio_dist": 1.0}
    dists = [p.dist for p in placements]
    return {
        "n": len(dists),
        "mean_tab_audio_dist": round(sum(dists) / len(dists), 3),
        "pct_close": round(100 * sum(d < 0.4 for d in dists) / len(dists), 1),
    }


def score_audio_vs_ireal(audio_seq, ireal_seq) -> dict:
    """Jaccard distances between audio-inferred chords and iReal GT.

    Uses the existing DTW alignment (align_tab_to_chart-style) but here
    the 'tab' is the audio sequence and the reference is iReal.
    Gives a baseline: how well does the Harmonia pipeline match iReal?
    """
    from harmonia.tab_aligner import _dtw, _best_transpose, _chord_dist

    if not audio_seq or not ireal_seq:
        return {"n": 0, "mean_audio_ireal_dist": 1.0}

    audio_pairs = [(a.pc, a.quality) for a in audio_seq]
    offset, _ = _best_transpose(ireal_seq, audio_pairs)
    transposed = [((pc + offset) % 12 if pc >= 0 else -1, q) for pc, q in audio_pairs]
    _, path = _dtw(ireal_seq, transposed)

    ireal_to_audio: dict[int, int] = {}
    for ii, ai in path:
        ireal_to_audio[ii] = ai

    dists = []
    for ii, (ir_pc, ir_q) in enumerate(ireal_seq):
        ai = ireal_to_audio.get(ii)
        if ai is None:
            dists.append(1.0)
            continue
        a_pc, a_q = transposed[ai]
        dists.append(_chord_dist(ir_pc, ir_q, a_pc, a_q))

    return {
        "n": len(dists),
        "mean_audio_ireal_dist": round(sum(dists) / len(dists), 3),
        "pct_close": round(100 * sum(d < 0.4 for d in dists) / len(dists), 1),
        "transpose": offset,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs",  type=int, default=20,
                    help="Max songs to evaluate (default 20)")
    ap.add_argument("--delay",  type=float, default=1.5,
                    help="Seconds between UG requests")
    ap.add_argument("--out",    type=Path,
                    default=REPO / "docs" / "tab_alignment_audio_eval.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--min-tab-rating", type=float, default=3.5)
    args = ap.parse_args()

    audio_cache = REPO / "data" / "cache" / "yt_audio"
    chart_cache = REPO / "data" / "cache" / "yt_charts"
    audio_cache.mkdir(parents=True, exist_ok=True)
    chart_cache.mkdir(parents=True, exist_ok=True)

    from harmonia.tab_fetcher import search_tabs, fetch_tab_chords

    existing: dict[str, dict] = {}
    if args.resume and args.out.exists():
        existing = {r["title"]: r for r in json.loads(args.out.read_text())}
        print(f"Resuming: {len(existing)} already done.")

    tunes = load_playlist(REPO / "data" / "ireal" / "pop400.txt")
    tunes = [t for t in tunes if t.time_signature and t.time_signature[0] == 4]
    print(f"4/4 pop songs available: {len(tunes)}")

    results: list[dict] = []
    n_done = 0

    for tune in tunes:
        if n_done >= args.songs:
            break

        title    = tune.title
        composer = tune.composer or ""
        artist   = re.sub(r"\s*\(.*?\)", "", composer).strip()

        if args.resume and title in existing:
            results.append(existing[title])
            n_done += 1
            continue

        # iReal ground truth
        try:
            mma_chart = tune_to_mma(tune)
        except Exception as e:
            print(f"  SKIP {title!r}: MMA parse error: {e}")
            continue

        ireal_seq = ireal_to_flat_seq(mma_chart)
        if len(ireal_seq) < 20:
            print(f"  SKIP {title!r}: only {len(ireal_seq)} GT chords")
            continue

        n_done += 1
        print(f"\n[{n_done:3d}] {title!r}  ({artist})  [{len(ireal_seq)} iReal chords]")

        # ── Step 1: YouTube audio ──────────────────────────────────────────
        wav_path = download_audio(title, artist, audio_cache)
        if wav_path is None:
            print(f"       no audio downloaded")
            results.append({"title": title, "composer": composer, "status": "no_audio"})
            continue
        time.sleep(1.0)   # be nice to YouTube

        # ── Step 2: Harmonia pipeline ──────────────────────────────────────
        chart_dict = run_harmonia(wav_path, chart_cache)
        if chart_dict is None:
            results.append({"title": title, "composer": composer, "status": "pipeline_error"})
            continue

        # Build a minimal ChordChart-like object from the cached JSON
        class _Chart:
            pass
        audio_chart = _Chart()
        audio_chart.chords = chart_dict["chords"]
        audio_chart.tempo_bpm = chart_dict["tempo_bpm"]
        audio_chart.time_signature = chart_dict["time_signature"]
        audio_chart.duration_s = chart_dict["duration_s"]

        audio_seq = audio_chart_to_sequence(audio_chart)
        print(f"       audio: {len(audio_seq)} chord events, "
              f"{audio_chart.tempo_bpm:.0f} BPM, {audio_chart.duration_s:.0f}s")

        # Score audio vs iReal (pipeline quality baseline)
        audio_ireal = score_audio_vs_ireal(audio_seq, ireal_seq)
        print(f"       audio↔iReal: dist={audio_ireal['mean_audio_ireal_dist']:.3f}  "
              f"close={audio_ireal['pct_close']}%")

        # ── Step 3: UG tab ─────────────────────────────────────────────────
        try:
            tab_results = search_tabs(title, artist)
            time.sleep(args.delay)
        except Exception as e:
            print(f"       search failed: {e}")
            results.append({"title": title, "composer": composer, "status": "search_error",
                             "audio_ireal": audio_ireal})
            continue

        title_matched = [r for r in tab_results
                         if _title_match(title, r.song_name)
                         and _artist_match(composer, r.artist_name)]

        if not title_matched:
            def _why(r):
                t = "T✓" if _title_match(title, r.song_name) else f"T✗({_canon_title(r.song_name)!r})"
                a = "A✓" if _artist_match(composer, r.artist_name) else f"A✗({r.artist_name!r})"
                return f"{r.song_name!r} {t} {a}"
            print(f"       no tab — {' | '.join(_why(r) for r in tab_results[:3])}")
            results.append({"title": title, "composer": composer, "status": "no_tab",
                             "audio_ireal": audio_ireal})
            continue

        good = [r for r in title_matched
                if r.rating >= args.min_tab_rating and r.tab_type in ("Chords", "Tab")]
        best = max(good or title_matched, key=lambda r: r.rating * (1 + 0.1 * min(r.votes, 500)))
        print(f"       tab: {best.rating:.2f}★ ({best.votes}v) — {best.tab_type}")

        try:
            tab = fetch_tab_chords(best)
            time.sleep(args.delay)
        except Exception as e:
            print(f"       fetch failed: {e}")
            results.append({"title": title, "composer": composer, "status": "fetch_error",
                             "audio_ireal": audio_ireal})
            continue

        if tab is None:
            results.append({"title": title, "composer": composer, "status": "fetch_none",
                             "audio_ireal": audio_ireal})
            continue

        tab_tokens = tab_to_chord_tokens(tab.raw_content, bpb=mma_chart.beats_per_bar)
        if not tab_tokens:
            results.append({"title": title, "composer": composer, "status": "empty_tab",
                             "audio_ireal": audio_ireal})
            continue

        print(f"       tab chords: {len(tab_tokens)}")

        # ── Step 4: Align tab to audio ──────────────────────────────────────
        try:
            audio_result = align_tab_to_audio(audio_seq, tab_tokens)
        except Exception as e:
            print(f"       audio alignment error: {e}")
            results.append({"title": title, "composer": composer, "status": "align_error",
                             "audio_ireal": audio_ireal})
            continue

        tab_audio = score_placements(audio_result.placements)
        print(f"       tab↔audio: dist={tab_audio['mean_tab_audio_dist']:.3f}  "
              f"close={tab_audio['pct_close']}%  transpose={audio_result.transpose_semitones}st")

        # ── Step 5: Also run old iReal-direct alignment for comparison ──────
        from harmonia.tab_aligner import align_tab_to_chart
        ireal_payload = [{"root": pc, "lv": {"seventh": {"q": q}}} for pc, q in ireal_seq]
        try:
            ireal_result = align_tab_to_chart(
                ireal_payload, tab_tokens,
                tab_rating=best.rating, tab_votes=best.votes,
            )
            ireal_direct = {
                "exact_pct": round(100 * sum(a.match == "exact" for a in ireal_result.annotations)
                                   / max(len(ireal_result.annotations), 1), 1),
                "mean_dist": round(sum(a.dist for a in ireal_result.annotations)
                                   / max(len(ireal_result.annotations), 1), 3),
                "transpose": ireal_result.transpose_semitones,
            }
        except Exception:
            ireal_direct = {}

        results.append({
            "title":            title,
            "composer":         composer,
            "status":           "ok",
            "ireal_chords":     len(ireal_seq),
            "audio_chords":     len(audio_seq),
            "tab_chords":       len(tab_tokens),
            "tempo_bpm":        audio_chart.tempo_bpm,
            "duration_s":       audio_chart.duration_s,
            "tab_rating":       round(best.rating, 3),
            "tab_votes":        best.votes,
            "audio_ireal":      audio_ireal,      # pipeline quality vs GT
            "tab_audio":        tab_audio,         # tab vs audio grid
            "ireal_direct":     ireal_direct,      # old method (tab vs iReal directly)
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    ok = [r for r in results if r.get("status") == "ok"]
    print(f"\n{'='*60}")
    print(f"Songs attempted: {n_done}  |  Fully aligned: {len(ok)}")
    if ok:
        def avg(key_path):
            keys = key_path.split(".")
            vals = []
            for r in ok:
                v = r
                for k in keys:
                    v = v.get(k, None)
                    if v is None:
                        break
                if v is not None:
                    vals.append(v)
            return round(sum(vals) / len(vals), 3) if vals else None

        print(f"\nMean over {len(ok)} songs:")
        print(f"  audio↔iReal dist : {avg('audio_ireal.mean_audio_ireal_dist')}  "
              f"(pipeline vs GT — lower = better Harmonia)")
        print(f"  tab↔audio dist   : {avg('tab_audio.mean_tab_audio_dist')}  "
              f"(tab vs audio grid)")
        print(f"  tab↔iReal direct : {avg('ireal_direct.mean_dist')}  "
              f"(old method, tab vs GT directly)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nResults → {args.out}")


if __name__ == "__main__":
    main()
