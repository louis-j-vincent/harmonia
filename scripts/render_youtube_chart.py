"""
render_youtube_chart.py — download a YouTube video and render an interactive chord chart.

Usage:
    .venv/bin/python scripts/render_youtube_chart.py https://youtu.be/XYZ
    .venv/bin/python scripts/render_youtube_chart.py https://youtu.be/XYZ --title "My Song"
    .venv/bin/python scripts/render_youtube_chart.py --audio song.wav  # skip download

Requires yt-dlp:
    pip install yt-dlp

Writes an HTML file (same interactive viewer as the existing inferred charts) and
opens it in the default browser.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# ── Harmonia label → iReal token mapping ─────────────────────────────────────
#
# ChordQuality.value strings (from chord_vocabulary.py) mapped to iReal tokens.
# iReal uses: "" = major triad, "-" = minor, "^7" = maj7, "-7" = min7, "7" = dom7,
# "o" = dim, "o7" = dim7, "h7" = half-dim, "+" = aug, "sus" = sus4, etc.

_QUALITY_TO_IREAL: dict[str, str] = {
    # Triads
    "maj":      "",
    "min":      "-",
    "dim":      "o",
    "aug":      "+",
    # Suspended
    "sus2":     "sus2",
    "sus4":     "sus",
    # Seventh chords
    "maj7":     "^7",
    "min7":     "-7",
    "7":        "7",
    "mMaj7":    "-^7",
    "ø7":       "h7",
    "°7":       "o7",
    "augMaj7":  "+^7",
    "aug7":     "+7",
    "7sus4":    "7sus",
    # sev_h spellings from chord_pipeline_v1.py (_SEV_TO_Q5) — this table was
    # originally written against chord_vocabulary.ChordQuality.value strings
    # (an older pipeline's vocabulary); chord_pipeline_v1's sev_h uses
    # different names for the same three qualities. Missing entries here
    # silently fell through .get(quality, quality) and printed the raw sev_h
    # string as the ireal token (e.g. "hdim7" instead of "h7").
    "hdim7":    "h7",
    "dim7":     "o7",
    "minmaj7":  "-^7",
    # Phase 2 — dominant altered / 9ths
    "9":        "9",
    "min9":     "-9",
    "maj9":     "^9",
    "7b9":      "7b9",
    "7#9":      "7#9",
    "9sus4":    "9sus",
    "maj9#11":  "^9#11",
    # Phase 3 — 11ths
    "min11":    "-11",
    "dom11":    "11",
    "maj11":    "^11",
    # Phase 4 — 13ths
    "dom13":    "13",
    "min13":    "-13",
    "maj13":    "^13",
    # Sixth chords (2026-07-21, iReal-import round-trip — see irealb_fetcher
    # ._IREAL_TOKEN_TO_SEV). "min6" has no entry-less fallback that works:
    # .get(quality, quality) would print the literal string "min6" as the
    # suffix instead of "-6".
    "6":        "6",
    "min6":     "-6",
    # No chord
    "N":        "N.C.",
}

# Collapse a full quality to seventh level (family + 7th, no extensions)
_QUALITY_TO_SEVENTH: dict[str, str] = {
    "9":        "7",
    "min9":     "min7",
    "maj9":     "maj7",
    "7b9":      "7",
    "7#9":      "7",
    "9sus4":    "7sus4",
    "maj9#11":  "maj7",
    "min11":    "min7",
    "dom11":    "7",
    "maj11":    "maj7",
    "dom13":    "7",
    "min13":    "min7",
    "maj13":    "maj7",
}

# Collapse any quality to family (triad only)
_QUALITY_TO_FAMILY: dict[str, str] = {
    "min7":     "min",
    "dom7":     "maj",  # dominant = major family
    "7":        "maj",
    "maj7":     "maj",
    "mMaj7":    "min",
    "ø7":       "min",
    "°7":       "dim",
    "augMaj7":  "aug",
    "aug7":     "aug",
    "7sus4":    "sus4",
    "sus2":     "sus2",
    "sus4":     "sus4",
    "hdim7":    "min",
    "dim7":     "dim",
    "minmaj7":  "min",
    "9":        "maj",
    "min9":     "min",
    "maj9":     "maj",
    "7b9":      "maj",
    "7#9":      "maj",
    "9sus4":    "sus4",
    "maj9#11":  "maj",
    "min11":    "min",
    "dom11":    "maj",
    "maj11":    "maj",
    "dom13":    "maj",
    "min13":    "min",
    "maj13":    "maj",
}

# Valid root names in Harmonia (SEMITONE_NAMES order)
_ROOTS = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}

# q5 family name (see chord_pipeline_v1._top_chord_suggestions) -> iReal token,
# at family/seventh granularity (no extensions — that's the depth the q5
# posterior actually distinguishes).
_Q5_TO_IREAL: dict[str, str] = {
    "maj": "", "min": "-", "dom": "7", "hdim": "h7", "dim": "o7",
}


def _split_label(label: str) -> tuple[str, str]:
    """Split "D:maj7" -> ("D", "maj7"), or "F#min7" -> ("F#", "min7") for
    callers still passing the older concatenated format. Returns ("", label)
    on failure.

    chord_pipeline_v1.infer_chords_v1 always emits the colon form
    (``f"{NOTE[root]}:{sev_h}"``) — this was previously unhandled here, so
    every quality token coming out of a *fresh* analysis carried a leading
    ":" (e.g. ireal token ":maj7" instead of "^7"), silently corrupting the
    chart display for any newly-analyzed song while every already-rendered
    chart (generated before this bug, or via a different pipeline) looked
    fine. Caught 2026-07-12 while adding chord-suggestion data, which flows
    through this same function.
    """
    if ":" in label:
        root, _, quality = label.partition(":")
        if root in _ROOTS:
            return root, quality
    if len(label) >= 2 and label[:2] in _ROOTS:
        return label[:2], label[2:]
    if label[:1] in _ROOTS:
        return label[:1], label[1:]
    return "", label


def _quality_ireal(quality: str, level: str) -> str:
    """Convert a Harmonia quality string to an iReal token at the given level.

    level: "exact" | "seventh" | "family"
    """
    if level == "seventh":
        quality = _QUALITY_TO_SEVENTH.get(quality, quality)
    elif level == "family":
        quality = _QUALITY_TO_FAMILY.get(quality, quality)
        quality = _QUALITY_TO_SEVENTH.get(quality, quality)  # two-step collapse
    return _QUALITY_TO_IREAL.get(quality, quality)


def label_to_ireal(label: str, level: str = "exact") -> str:
    """Convert a Harmonia chord label to an iReal token at the given depth.

    A slash-bass label ("Bb:maj/D", the format irealb_fetcher.py emits for a
    slash chord) must have its "/bass" stripped BEFORE the quality lookup —
    "maj/D" isn't a key in _QUALITY_TO_IREAL and fell through unmapped,
    printing "/D" glued onto whatever the unmatched-quality fallback produced
    (confirmed bug 2026-07-21, "les slash chords se perdent"). Re-attach the
    bass note untouched at every level — it's a real, sounding note, not an
    extension "family"/"seventh" collapse should ever hide.
    """
    root, quality = _split_label(label)
    if not root:
        return label
    quality, _, bass = quality.partition("/")
    token = root + _quality_ireal(quality, level)
    return token + ("/" + bass if bass else "")


# ── Audio download ────────────────────────────────────────────────────────────

def _check_ytdlp() -> None:
    if shutil.which("yt-dlp") is None:
        print("Error: yt-dlp not found.  Install with:  pip install yt-dlp", file=sys.stderr)
        sys.exit(1)


def _download_audio(url: str, dest_dir: Path, verbose: bool) -> tuple[Path, str]:
    """Download audio from a YouTube URL. Returns (audio_path, video_title)."""
    template = str(dest_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "best",
        "--audio-quality", "0",
        "-o", template,
        "--print", "after_move:filepath",
        "--print", "title",
        "--no-playlist",
        url,
    ]
    if not verbose:
        cmd += ["--quiet", "--no-warnings"]

    log = logging.getLogger(__name__)
    log.info("Downloading audio from %s", url)
    result = subprocess.run(cmd, capture_output=not verbose, text=True, check=False)

    if result.returncode != 0:
        print(f"Error: yt-dlp failed (exit {result.returncode})\n{result.stderr or ''}",
              file=sys.stderr)
        sys.exit(1)

    lines = (result.stdout or "").strip().splitlines()
    title = ""
    audio_path: Path | None = None

    # yt-dlp prints filepath then title (one --print per line)
    for line in lines:
        p = Path(line)
        if p.exists():
            audio_path = p
        elif not title:
            title = line

    if audio_path is None:
        # Fallback: newest audio file in dest_dir
        audio_exts = {".opus", ".m4a", ".mp3", ".webm", ".ogg", ".flac", ".wav"}
        files = sorted(dest_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for f in files:
            if f.suffix in audio_exts:
                audio_path = f
                break

    if audio_path is None:
        print("Error: could not locate downloaded audio file.", file=sys.stderr)
        sys.exit(1)

    return audio_path, title


# ── ChordChart → render_interactive adapter ───────────────────────────────────

def _parse_time_signature(ts_str: str) -> tuple[int, int]:
    """Parse "4/4" → (4, 4). Falls back to (4, 4)."""
    try:
        num, den = ts_str.split("/")
        return int(num), int(den)
    except Exception:
        return 4, 4


def rebalance_near_boundary_onsets(chord_dicts: list[dict], bpb: int) -> int:
    """Fix the "2 chords crammed into 1 bar, next bar shows a bare repeat" bug
    (docs/known_issues.md 2026-07-18 "★ CHART / BAR-GRID onset rebalance").

    Root cause: bar/beat is assigned per chord by ``floor(abs_beat / bpb)`` —
    a chord whose real onset lands just past a bar's midpoint gets floored
    into the CURRENT bar even though it musically belongs to (and mostly
    sounds through) the NEXT bar, which the sparse "start of each change"
    chord-list format then renders as empty/held ("%"). Corpus check on
    autumn_leaves found this is systematic, not a one-off: 11/329 bars hit it
    at the baked offset=0 grid, worsening to 17/328 after a user-applied
    global bar-1 phase correction (offset selected to fix the song's INTRO
    can't simultaneously be right for every later passage — same rigid-grid
    limitation as the "GRID PHASE MISALIGNMENT" structure-metric finding, now
    confirmed as the same phenomenon surfacing in the chart UI).

    Mitigation (targeted, not a full re-grid): when a bar ends up with >=2
    onsets AND the immediately following bar has none, and the LAST onset in
    that bar sits in its back half (``beat >= bpb/2``), that onset almost
    certainly belongs to the next bar instead — move it there (``bar += 1``,
    ``beat -= bpb``). Mutates ``chord_dicts`` in place (each item needs
    ``bar``/``beat`` keys); returns the number of onsets moved so callers can
    skip re-deriving ``n_bars``/sections when nothing changed. Deliberately
    narrow: bars with >=2 onsets where the next bar ALSO has content are left
    untouched (that's ordinary fast harmonic rhythm, not this failure mode).
    """
    by_bar: dict[int, list[int]] = {}
    for i, c in enumerate(chord_dicts):
        by_bar.setdefault(c["bar"], []).append(i)
    moved = 0
    for bar in sorted(by_bar):
        idxs = by_bar[bar]
        if len(idxs) < 2 or by_bar.get(bar + 1):
            continue
        last = max(idxs, key=lambda i: chord_dicts[i]["beat"])
        if chord_dicts[last]["beat"] >= bpb / 2.0:
            # Re-anchor as the (only) onset of the next bar. Not
            # ``beat - bpb`` — ``beat`` is already reduced mod bpb by the
            # caller (always in [0, bpb)), so that subtraction would always
            # go negative. The next bar was confirmed empty above, so beat=0
            # (this chord now leads it) can't collide with anything.
            chord_dicts[last]["bar"] = bar + 1
            chord_dicts[last]["beat"] = 0
            by_bar[bar].remove(last)
            by_bar.setdefault(bar + 1, []).append(last)
            moved += 1
    return moved


def chart_to_interactive_inputs(pipeline_chart, title: str, source_desc: str,
                                 bar1_offset_beats: int = 0):
    """Convert a ChordChart (from HarmoniaPipeline) to inputs for render_interactive.

    Returns (chart_obj, chord_dicts) where chord_dicts have the {bar, beat, levels}
    format that render_interactive expects.

    ``bar1_offset_beats``: shifts the PHASE of the bar grid (which detected
    beat counts as the first beat of bar 1) without touching its STEP size —
    the step size (real per-beat tempo, via start_beat_idx) was already fixed
    2026-07-15 (see docs/known_issues.md "Chart bar-layout bug"), but that fix
    left the phase/origin exactly where the raw beat tracker's beat 0 lands,
    which is not necessarily the actual downbeat (e.g. a pickup/anacrusis, or
    the tracker locking onto an off-beat accent). Positive N means "the true
    bar-1 downbeat is N beats after the tracker's beat 0" — those N beats
    become a pickup (clamped into bar 0, never negative). Set per-song via
    /bar1-offset-fix, stored in data/cache/chart_bar1_offsets.json.
    """
    from harmonia.output.chart_render import Chart

    bpb, _ = _parse_time_signature(pipeline_chart.time_signature)
    beat_dur_s = 60.0 / max(pipeline_chart.tempo_bpm, 1.0)

    # ── display-layer bar condensation (user directive 2026-07-19) ────────────
    # When the median chord spans ≥ ~2 bars the grid is a sea of held ("%") cells
    # — usually a 2× tempo octave-lock (documented UNSOLVABLE blind, known_issues
    # #1), e.g. abba Chiquitita analysed at 168 bpm / 225 bars / median 3.0
    # bars/chord.  The chord ONSET TIMES are correct regardless of the tempo
    # octave, so this is a DISPLAY concern, not an analysis one: fold the bar grid
    # 2× (or 4×) so a typical bar carries ~1 chord.  Done here (not chart_model)
    # because this is where seconds → (bar, beat) is laid out; halving the grid
    # beat here re-grids chords AND sections consistently.  NOT applied to
    # already-dense charts (henny/just-aint median 1.0 → factor 1, untouched).
    _bar_dur = bpb * beat_dur_s
    _durs = sorted((c["end_s"] - c["start_s"]) / max(_bar_dur, 1e-9)
                   for c in pipeline_chart.chords if c.get("label") != "N")
    condense = 1
    if _durs:
        _med = _durs[len(_durs) // 2]
        while _med / condense >= 1.75 and condense < 4:
            condense *= 2
    grid_beat_dur = beat_dur_s * condense          # condensed display beat

    # ── display-layer real-beat time snapping (user directive 2026-07-20) ──────
    # The decode grid is a UNIFORM bestfit period; it cannot absorb tempo rubato,
    # so a chord's uniform onset drifts from the real audio by the grid residual
    # (Let It Be: ±1.5 s vs real beats late in the song).  The (bar, beat) LAYOUT
    # stays on the uniform grid (computed above → stable sections/folds), but the
    # displayed t0/t1 that drive the PLAYHEAD are snapped to the nearest DETECTED
    # beat so the highlight tracks the audio.  Decode untouched.  No-op if the
    # pipeline carried no real beats.
    _rbeats = sorted(float(t) for t in getattr(pipeline_chart, "beat_times", []) or [])
    _rb = np.asarray(_rbeats) if _rbeats else None

    def _snap(t: float) -> float:
        if _rb is None or len(_rb) == 0:
            return t
        i = int(np.searchsorted(_rb, t))
        lo = _rb[i - 1] if i - 1 >= 0 else None
        hi = _rb[i] if i < len(_rb) else None
        if lo is not None and hi is not None:
            return float(lo if abs(lo - t) <= abs(hi - t) else hi)
        # t is OUTSIDE the tracked beat range entirely (the tracker lost lock
        # before the song ended, or before it began) — snapping every such t
        # to the single nearest edge beat collapses ALL of them onto that ONE
        # instant. Confirmed real bug, 2026-07-21 ("Hot n Cold": the beat
        # tracker stops at 195.9s in a 283s song — a whispered-bridge section
        # with almost no rhythmic onset content — and every chord after that
        # point froze to that exact same timestamp, breaking both playback
        # and the displayed 4-chord loop for the whole second half). Only
        # trust the edge beat within ~2 real beat periods of it; beyond that,
        # fall back to the un-snapped uniform-grid time — its drift is a
        # known, BOUNDED error (±1.5s late in a song, per the note above); an
        # unbounded frozen timestamp is far worse.
        edge = hi if lo is None else lo
        period = float(np.median(np.diff(_rb))) if len(_rb) > 1 else 0.0
        if period > 0 and abs(t - edge) <= 2 * period:
            return float(edge)
        return t
    off_c = round(bar1_offset_beats / condense)    # offset in condensed beats

    chord_dicts = []
    for ch in pipeline_chart.chords:
        # Prefer the real detected-beat index (billboard_v1 backend; see
        # chord_pipeline_v1.infer_chords_billboard_v1's "start_beat_idx"
        # comment) over reconstructing a beat position from start_s /
        # (60/tempo_bpm). The latter re-lays a rigid constant-tempo grid over
        # the whole track — on real (non-metronomic) audio this drifts out of
        # sync with the music as the song progresses, producing bar numbers /
        # chord-density-per-bar that don't match the actual harmonic rhythm
        # (2026-07-15 bug report: "bar 26 at 0:34" into a 2:40 song).
        # infer_chords_v1 (POP909-tuned fallback) doesn't emit this field, so
        # fall back to the old time/tempo reconstruction there — its audio is
        # near-metronomic by construction, so the reconstruction is safe.
        if "start_beat_idx" in ch:
            abs_beat = int(round(ch["start_beat_idx"] / condense))
        else:
            abs_beat = int(ch["start_s"] / grid_beat_dur)
        # Do NOT clamp eff_beat to 0 before dividing: Python's floor // and %
        # already give a correct, collision-free (bar, beat) pair for a
        # negative eff_beat (pickup chords) — e.g. bpb=4, eff_beat=-1 ->
        # bar=-1, beat=3. Clamping eff_beat itself first would instead pin
        # EVERY pickup chord to the same (bar=0, beat=0), silently colliding
        # in the annotation sidecar's (bar, beat) correction key. Only the
        # final bar index is clamped to 0 (bars can't render negative).
        eff_beat = abs_beat - off_c
        bar = max(0, eff_beat // bpb)
        beat = eff_beat % bpb
        conf = float(ch["confidence"])
        label = ch["label"]

        # Synthesise the three-level breakdown from the single pipeline prediction.
        # All levels share the same confidence — the UI's Auto mode will just show
        # the exact level when confidence >= threshold (default 0.6).
        ireal_exact   = label_to_ireal(label, "exact")
        ireal_seventh = label_to_ireal(label, "seventh")
        ireal_family  = label_to_ireal(label, "family")

        suggestions = [
            {"root": s["root"], "ireal": _Q5_TO_IREAL.get(s["q5"], ""), "conf": s["prob"]}
            for s in ch.get("suggestions", [])
        ]

        # Playhead onset: prefer the trusted music-x-lab change-time (``onset_s``/
        # ``offset_s``, attached by _attach_musx_onset_hints) over the uniform
        # bar-grid time before snapping — the uniform onset can drift a full beat
        # off the real downbeat and snap to the WRONG detected beat (user report
        # 2026-07-20, This Love bar-1 highlighted "celui d'après").  The (bar,
        # beat) LAYOUT above is unchanged (uniform grid), so sections/folds are
        # byte-identical; only the highlighted time changes.
        _disp_t0 = ch.get("onset_s", ch["start_s"])
        _disp_t1 = ch.get("offset_s", ch["end_s"])
        chord_dicts.append({
            "bar": bar,
            "beat": beat,
            "start_s": _snap(_disp_t0),        # playhead time snapped to real beat
            "end_s": _snap(_disp_t1),
            "start_s_grid": ch["start_s"],      # uniform onset (audit / re-decode)
            "levels": {
                "family":  {"ireal": ireal_family,  "conf": conf},
                "seventh": {"ireal": ireal_seventh, "conf": conf},
                "exact":   {"ireal": ireal_exact,   "conf": conf},
            },
            "suggestions": suggestions,
        })

    rebalance_near_boundary_onsets(chord_dicts, bpb)
    n_bars = max((c["bar"] for c in chord_dicts), default=0) + 1

    # section_per_bar: bars with the same consecutive value form one section block.
    # We label every bar with its segment key so the interactive chart groups them
    # correctly (the renderer shows a section-start marker only when the label changes).
    section_per_bar = [""] * n_bars
    for seg in pipeline_chart.segments:
        # Same real-beat-index preference as the chord loop above; see its
        # comment. Falls back to time/tempo reconstruction for backends
        # (infer_chords_v1) that don't emit start_beat_idx.
        if "start_beat_idx" in seg:
            start_beat = int(round(seg["start_beat_idx"] / condense))
            end_beat = start_beat + int(round(int(seg.get("n_beats", 1)) / condense))
        else:
            start_beat = int(seg["start_s"] / grid_beat_dur)
            end_beat = int(seg["end_s"] / grid_beat_dur)
        # Same unclamped-then-floor pattern as the chord loop above (clamp
        # the bar index, not eff_beat itself). Condensed grid → condensed offset.
        start_bar = max(0, (start_beat - off_c) // bpb)
        end_bar   = min(max(0, (end_beat - off_c) // bpb) + 1, n_bars)
        key_tag   = seg.get("key", "")
        for b in range(start_bar, end_bar):
            section_per_bar[b] = key_tag

    chart_obj = Chart(
        title=title,
        composer="",
        key=pipeline_chart.global_key,
        style=source_desc,
        tempo=int(pipeline_chart.tempo_bpm),
        time_signature=(bpb, 4),
        n_bars=n_bars,
        section_per_bar=section_per_bar,
    )
    return chart_obj, chord_dicts


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download a YouTube video and render an interactive Harmonia chord chart.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("url", nargs="?", default=None, help="YouTube URL")
    src.add_argument("--audio", type=Path, default=None,
                     help="Analyse a local audio file instead of downloading")

    ap.add_argument("--title", default=None,
                    help="Chart title (default: video title or filename)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output HTML path (default: <title>_chart.html in cwd)")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache"),
                    help="Cache dir for Basic Pitch activations")
    ap.add_argument("--theta", type=float, default=0.08,
                    help="Cosine novelty threshold for chord-change detection (default 0.08)")
    ap.add_argument("--cols", type=int, default=4,
                    help="Bars per row in the chart")
    ap.add_argument("--keep-audio", action="store_true",
                    help="Keep downloaded audio after analysis")
    ap.add_argument("--no-open", action="store_true",
                    help="Don't open the HTML in the browser")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    # ── Acquire audio ─────────────────────────────────────────────────────────
    tmp_dir: Path | None = None
    video_title = ""

    if args.audio:
        if not args.audio.exists():
            sys.exit(f"Error: {args.audio} not found")
        audio_path = args.audio
        video_title = args.audio.stem
        source_desc = f"inferred from {args.audio.name}"
    else:
        _check_ytdlp()
        tmp_dir = Path(tempfile.mkdtemp(prefix="harmonia_yt_"))
        try:
            audio_path, video_title = _download_audio(args.url, tmp_dir, args.verbose)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        source_desc = f"inferred from YouTube · {args.url}"
        logging.getLogger(__name__).info("Audio: %s  title: %s", audio_path, video_title)

    title = args.title or video_title or audio_path.stem

    # ── Run pipeline (Gen-2 v1) ───────────────────────────────────────────────
    try:
        from harmonia.models.chord_pipeline_v1 import infer_chords_v1

        logging.getLogger(__name__).info("Running v1 pipeline on %s…", audio_path.name)
        pipeline_chart = infer_chords_v1(
            audio_path,
            seventh_gate=0.0,
            cache_dir=args.cache_dir,
        )
        pipeline_chart.print()

    finally:
        if tmp_dir and not args.keep_audio:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Convert and render ────────────────────────────────────────────────────
    from harmonia.output.chart_interactive import render_interactive

    chart_obj, chord_dicts = chart_to_interactive_inputs(pipeline_chart, title, source_desc)

    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "chart"
    out = args.out or Path(f"{slug}_chart.html")
    render_interactive(chart_obj, chord_dicts, out, bars_per_row=args.cols,
                       sections=pipeline_chart.sections)
    print(f"→ {out.resolve()}")

    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
