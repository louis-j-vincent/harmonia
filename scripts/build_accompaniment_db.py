"""Build the (song_structure, chords, bass, midi) accompaniment database.

Pipeline per song (validated end-to-end 2026-07-03):
    iReal chart → pyRealParser → MMAChart (harmonia.data.ireal_corpus)
    → MMA renders multi-track MIDI (named tracks: Bass, Chord, Drum, …)
    → bass track extracted by name, ground-truth chord timeline attached
    → one JSON record per song in db.jsonl + one .mid + one .mma file.

Usage:
    .venv/bin/python scripts/build_accompaniment_db.py                 # all corpora
    .venv/bin/python scripts/build_accompaniment_db.py --max-songs 20  # quick sample
    .venv/bin/python scripts/build_accompaniment_db.py --corpus jazz1460

Prerequisite: bash scripts/fetch_accompaniment_deps.sh
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pretty_midi  # noqa: E402

from harmonia.data.ireal_corpus import (  # noqa: E402
    chord_root_pc,
    load_playlist,
    tune_to_mma,
)

MMA_VERSION = "25.05.3"
DEFAULT_MMA_DIR = REPO / "data" / "tools" / f"mma-bin-{MMA_VERSION}"
DEFAULT_IREAL_DIR = REPO / "data" / "ireal"
DEFAULT_OUT_DIR = REPO / "data" / "accomp_db"

BASS_TRACK_HINTS = ("bass", "walk")


def load_mma_qualities(mma_dir: Path) -> set[str]:
    sys.path.insert(0, str(mma_dir))
    from MMA import chordtable  # noqa: PLC0415

    return set(chordtable.chordlist.keys())


def render_mma(mma_dir: Path, mma_path: Path, mid_path: Path) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, str(mma_dir / "mma.py"), str(mma_path), "-f", str(mid_path)],
        capture_output=True,
        text=True,
        cwd=str(mma_dir),
        timeout=120,
    )
    return mid_path.exists(), (result.stdout + result.stderr)[-2000:]


def extract_tracks(mid_path: Path) -> tuple[dict, list, float]:
    """Return ({track_name: n_notes}, bass note events, midi duration)."""
    pm = pretty_midi.PrettyMIDI(str(mid_path))
    tracks = {}
    bass_notes: list[list] = []
    for inst in pm.instruments:
        name = inst.name.strip() or ("Drum" if inst.is_drum else f"prog{inst.program}")
        tracks[name] = len(inst.notes)
        if not inst.is_drum and any(h in name.lower() for h in BASS_TRACK_HINTS):
            for n in inst.notes:
                bass_notes.append(
                    [n.pitch, round(n.start, 4), round(n.end, 4), n.velocity]
                )
    bass_notes.sort(key=lambda x: x[1])
    return tracks, bass_notes, pm.get_end_time()


def bass_root_agreement(chart, bass_notes) -> float | None:
    """Sanity metric: fraction of bar-initial bass notes matching the bar's chord root.

    Tolerance of ±15% of a beat around the barline absorbs groove humanization.
    """
    if not bass_notes:
        return None
    sec_per_beat = 60.0 / chart.tempo
    sec_per_bar = sec_per_beat * chart.beats_per_bar
    roots = {}
    for barno, _, slots in chart.timeline:
        pc = chord_root_pc(slots[0][2])
        if pc is not None:
            roots[barno] = pc
    hits = total = 0
    for pitch, start, _, _ in bass_notes:
        bar = int(start / sec_per_bar) + 1
        offset = start - (bar - 1) * sec_per_bar
        if bar in roots and offset < 0.15 * sec_per_beat:
            total += 1
            hits += pitch % 12 == roots[bar]
    return hits / total if total else None


def process_corpus(
    corpus_path: Path,
    out_dir: Path,
    mma_dir: Path,
    qualities: set[str],
    max_songs: int | None,
    db_file,
) -> dict:
    corpus = corpus_path.stem
    midi_dir = out_dir / "midi" / corpus
    mma_out_dir = out_dir / "mma" / corpus
    midi_dir.mkdir(parents=True, exist_ok=True)
    mma_out_dir.mkdir(parents=True, exist_ok=True)

    tunes = load_playlist(corpus_path)
    if max_songs:
        tunes = tunes[:max_songs]
    stats = {"corpus": corpus, "total": len(tunes), "rendered": 0, "skipped": 0,
             "unmapped_tokens": {}, "agreements": []}

    for idx, tune in enumerate(tunes):
        song_id = f"{corpus}_{idx:04d}"
        slug = re.sub(r"\W+", "_", tune.title).strip("_")[:60]
        try:
            chart = tune_to_mma(tune, valid_qualities=qualities)
        except Exception as exc:  # malformed chart
            stats["skipped"] += 1
            print(f"  SKIP {song_id} {tune.title!r}: {exc}")
            continue
        for tok in chart.unmapped_tokens:
            stats["unmapped_tokens"][tok] = stats["unmapped_tokens"].get(tok, 0) + 1
        if not chart.timeline:
            stats["skipped"] += 1
            continue

        mma_path = mma_out_dir / f"{song_id}_{slug}.mma"
        mid_path = midi_dir / f"{song_id}_{slug}.mid"
        mma_path.write_text(chart.mma_text)
        ok, log = render_mma(mma_dir, mma_path, mid_path)
        if not ok:
            stats["skipped"] += 1
            print(f"  FAIL {song_id} {tune.title!r}: {log[-200:]}")
            continue

        tracks, bass_notes, duration = extract_tracks(mid_path)
        agreement = bass_root_agreement(chart, bass_notes)
        if agreement is not None:
            stats["agreements"].append(agreement)

        sec_per_beat = 60.0 / chart.tempo
        chord_timeline = [
            {
                "bar": barno,
                "beat": beat,
                "time": round(((barno - 1) * chart.beats_per_bar + beat) * sec_per_beat, 4),
                "ireal": tok,
                "mma": chord,
            }
            for barno, _, slots in chart.timeline
            for beat, tok, chord in slots
        ]
        record = {
            "song_id": song_id,
            "corpus": corpus,
            "title": chart.title,
            "composer": chart.composer,
            "style": chart.style,
            "key": chart.key,
            "time_signature": list(chart.time_signature),
            "groove": chart.groove,
            "tempo": chart.tempo,
            "beats_per_bar": chart.beats_per_bar,
            "n_bars": len(chart.timeline),
            "form": chart.form,
            "section_per_bar": chart.section_per_bar,
            "chord_timeline": chord_timeline,
            "bass_notes": bass_notes,
            "bass_root_agreement": agreement,
            "tracks": tracks,
            "duration_sec": round(duration, 2),
            "midi_path": str(mid_path.relative_to(out_dir.parent.parent)),
            "mma_path": str(mma_path.relative_to(out_dir.parent.parent)),
            "unmapped_tokens": chart.unmapped_tokens,
        }
        db_file.write(json.dumps(record) + "\n")
        stats["rendered"] += 1
        if stats["rendered"] % 100 == 0:
            print(f"  … {stats['rendered']}/{len(tunes)} rendered")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", action="append",
                    help="corpus stem(s) under data/ireal (default: all .txt files)")
    ap.add_argument("--max-songs", type=int, default=None)
    ap.add_argument("--ireal-dir", type=Path, default=DEFAULT_IREAL_DIR)
    ap.add_argument("--mma-dir", type=Path, default=DEFAULT_MMA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    if not args.mma_dir.exists():
        sys.exit(f"MMA not found at {args.mma_dir} — run scripts/fetch_accompaniment_deps.sh")
    corpora = (
        [args.ireal_dir / f"{c}.txt" if not (args.ireal_dir / c).exists() else args.ireal_dir / c
         for c in args.corpus]
        if args.corpus
        else sorted(args.ireal_dir.glob("*.txt")) + sorted(args.ireal_dir.glob("*.irealb"))
    )
    if not corpora:
        sys.exit(f"No corpora in {args.ireal_dir} — run scripts/fetch_accompaniment_deps.sh")

    qualities = load_mma_qualities(args.mma_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.out_dir / "db.jsonl"

    t0 = time.time()
    all_stats = []
    with open(db_path, "w") as db_file:
        for corpus_path in corpora:
            print(f"== {corpus_path.stem}")
            all_stats.append(
                process_corpus(corpus_path, args.out_dir, args.mma_dir,
                               qualities, args.max_songs, db_file)
            )

    print(f"\n== Summary ({time.time() - t0:.0f}s) → {db_path}")
    for s in all_stats:
        agr = s["agreements"]
        mean_agr = sum(agr) / len(agr) if agr else float("nan")
        print(f"  {s['corpus']}: {s['rendered']}/{s['total']} rendered, "
              f"{s['skipped']} skipped, mean bass-root agreement {mean_agr:.1%}")
        if s["unmapped_tokens"]:
            top = sorted(s["unmapped_tokens"].items(), key=lambda kv: -kv[1])[:10]
            print(f"    unmapped tokens: {top}")


if __name__ == "__main__":
    main()
