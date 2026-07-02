"""
Cross-song empirical analysis of bass-note patterns vs ground-truth chord
changes -- the observational groundwork for the bass-pattern hypothesis
discussed 2026-07 (see docs/known_issues.md #1, scripts/bass_track.py, and
scripts/plot_bass_and_key_tracks.py for the per-song visual version this
aggregates across songs). Nothing here is wired into the pipeline yet --
this is exploratory, learning empirical distributions from real data before
deciding what (if anything) to build on top of them.

The hypothesis, concretely: a walking bass line moves every beat or two
without necessarily implying a new chord (e.g. 1-5-7 over a static chord);
a bass pitch-class change that coincides with other evidence changing is
more likely to be a genuine chord change. This script checks the two halves
of that directly against real ground truth:
  1. bass_scale_degree_hist.png -- when there IS a real GT chord sounding,
     what scale degree (relative to the chord root) does the bass tend to
     sit on? Confirms/refutes "bass favours root and fifth."
  2. bass_chord_change_correlation.png -- P(chord actually changed | bass
     pitch class changed) vs P(chord changed | bass pitch class same). If
     bass-change predicts chord-change, these should differ a lot.
  3. run_length_comparison.png -- distribution of how many consecutive
     beats the bass pitch class stays put vs how many consecutive beats the
     GT chord stays the same. Tests "bass moves faster than harmony" directly.

Usage:
    .venv/bin/python scripts/analyze_bass_patterns.py --songs 001 002 003 004 005
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from bass_track import compress_to_runs, forward_fill_bass, infer_bass_track_learned  # noqa: E402

DATA_ROOT = Path(__file__).parent.parent / "data"
PLOT_ROOT = Path(__file__).parent.parent / "docs" / "plots" / "inference" / "bass_patterns"
MIDI_START = 21
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def _gt_chord_at_time(gt_chords, t: float):
    """Returns (root, quality_value) for the GT chord active at time t, or None."""
    for ev in gt_chords:
        if ev.start_beat <= t < ev.end_beat:
            if ev.root < 0:
                return None  # N / no-chord
            return (ev.root, ev.quality.value)
    return None


def collect_song_data(song_id: str) -> dict | None:
    from harmonia.data.pop909_parser import POP909Parser
    from harmonia.models.rhythm import RhythmAnalyser
    from harmonia.models.stage1_pitch import PitchExtractor

    wav = DATA_ROOT / "renders" / "pop909" / song_id / f"{song_id}_v005_musescoregeneral.wav"
    if not wav.exists():
        print(f"  {song_id}: no wav, skipping")
        return None

    gt_song = POP909Parser(DATA_ROOT / "pop909" / "POP909").parse_song(song_id)
    if gt_song is None or not gt_song.chord_events:
        print(f"  {song_id}: no GT, skipping")
        return None

    extractor = PitchExtractor(cache_dir=DATA_ROOT / "cache")
    rhythm = RhythmAnalyser(prefer_madmom=False)
    act = extractor.extract(wav)
    bg = rhythm.analyse(wav)
    beat_probs_onset = bg.quantise_frames(act.frame_times, act.onset_probs)
    B = beat_probs_onset.shape[0]

    bass_raw = infer_bass_track_learned(beat_probs_onset)
    bass = forward_fill_bass(bass_raw)

    gt_root = np.full(B, -2, dtype=int)     # -2 = no GT coverage, -1 = N, >=0 = real root
    gt_quality = [None] * B
    for b in range(B):
        r = _gt_chord_at_time(gt_song.chord_events, bg.beat_times[b])
        if r is None:
            gt_root[b] = -1
        else:
            gt_root[b], gt_quality[b] = r

    # Chord run lengths (beats), straight from the GT events themselves --
    # each event already IS one contiguous chord span by construction.
    chord_run_beats = []
    for ev in gt_song.chord_events:
        if ev.root < 0:
            continue
        b_start = int(np.searchsorted(bg.beat_times, ev.start_beat, side="left"))
        b_end = int(np.searchsorted(bg.beat_times, ev.end_beat, side="left"))
        n = max(0, min(b_end, B) - min(b_start, B))
        if n > 0:
            chord_run_beats.append(n)

    return {
        "song_id": song_id, "B": B, "bass": bass, "bass_raw": bass_raw,
        "gt_root": gt_root, "gt_quality": gt_quality,
        "chord_run_beats": chord_run_beats,
    }


# ---------------------------------------------------------------------------
# 1. Bass scale-degree histogram
# ---------------------------------------------------------------------------

def compute_scale_degree_hist(all_data: list[dict]) -> np.ndarray:
    counts = np.zeros(12, dtype=int)
    for d in all_data:
        bass, gt_root = d["bass"], d["gt_root"]
        for b in range(d["B"]):
            if bass[b] < 0 or gt_root[b] < 0:
                continue
            interval = (bass[b] - gt_root[b]) % 12
            counts[interval] += 1
    return counts


def plot_scale_degree_hist(counts: np.ndarray, n_songs: int) -> None:
    degree_names = ["root(0)", "b2(1)", "2(2)", "b3(3)", "3(4)", "4(5)",
                     "b5(6)", "5(7)", "b6(8)", "6(9)", "b7(10)", "7(11)"]
    fig, ax = plt.subplots(figsize=(9, 5))
    total = counts.sum()
    frac = counts / max(total, 1)
    colors = ["#2ca02c" if i in (0, 7) else ("#1f77b4" if i == 4 else "#888888") for i in range(12)]
    ax.bar(degree_names, frac, color=colors)
    for i, f in enumerate(frac):
        ax.text(i, f + 0.005, f"{f:.1%}", ha="center", fontsize=8)
    ax.set_ylabel("share of (bass, GT-chord) beat observations")
    ax.set_title(
        f"Inferred bass note's scale degree relative to concurrent GT chord root\n"
        f"pooled across {n_songs} songs, n={total} beats "
        f"(green=root/fifth, blue=third, grey=other)"
    )
    fig.tight_layout()
    out = PLOT_ROOT / "bass_scale_degree_hist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}  (root={frac[0]:.1%}, fifth={frac[7]:.1%}, third={frac[4]:.1%})")


# ---------------------------------------------------------------------------
# 2. Bass-change vs chord-change correlation
# ---------------------------------------------------------------------------

def compute_change_contingency(all_data: list[dict]) -> dict:
    n_bc_cc = n_bc_nc = n_nc_cc = n_nc_nc = 0
    for d in all_data:
        bass, gt_root, gt_quality = d["bass"], d["gt_root"], d["gt_quality"]
        for b in range(1, d["B"]):
            if bass[b] < 0 or bass[b - 1] < 0:
                continue
            if gt_root[b] < 0 or gt_root[b - 1] < 0:  # need a real chord at both beats
                continue
            bass_changed = (bass[b] % 12) != (bass[b - 1] % 12)
            chord_changed = (gt_root[b], gt_quality[b]) != (gt_root[b - 1], gt_quality[b - 1])
            if bass_changed and chord_changed:
                n_bc_cc += 1
            elif bass_changed and not chord_changed:
                n_bc_nc += 1
            elif not bass_changed and chord_changed:
                n_nc_cc += 1
            else:
                n_nc_nc += 1
    return {"bc_cc": n_bc_cc, "bc_nc": n_bc_nc, "nc_cc": n_nc_cc, "nc_nc": n_nc_nc}


def plot_change_contingency(c: dict) -> None:
    n_bass_changed = c["bc_cc"] + c["bc_nc"]
    n_bass_same = c["nc_cc"] + c["nc_nc"]
    n_chord_changed = c["bc_cc"] + c["nc_cc"]
    n_chord_same = c["bc_nc"] + c["nc_nc"]

    p_chord_change_given_bass_change = c["bc_cc"] / max(n_bass_changed, 1)
    p_chord_change_given_bass_same = c["nc_cc"] / max(n_bass_same, 1)
    p_bass_change_given_chord_change = c["bc_cc"] / max(n_chord_changed, 1)
    p_bass_change_given_chord_same = c["bc_nc"] / max(n_chord_same, 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    bars = ax.bar(["bass changed\n(n={})".format(n_bass_changed),
                    "bass same\n(n={})".format(n_bass_same)],
                   [p_chord_change_given_bass_change, p_chord_change_given_bass_same],
                   color=["#d62728", "#1f77b4"])
    for bar, v in zip(bars, [p_chord_change_given_bass_change, p_chord_change_given_bass_same]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.1%}", ha="center")
    ax.set_ylabel("P(GT chord changed)")
    ax.set_title("Does a bass pitch-class change predict a real chord change?")
    ax.set_ylim(0, 1)

    ax = axes[1]
    bars = ax.bar(["chord changed\n(n={})".format(n_chord_changed),
                    "chord same\n(n={})".format(n_chord_same)],
                   [p_bass_change_given_chord_change, p_bass_change_given_chord_same],
                   color=["#d62728", "#1f77b4"])
    for bar, v in zip(bars, [p_bass_change_given_chord_change, p_bass_change_given_chord_same]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.1%}", ha="center")
    ax.set_ylabel("P(bass pitch class changed)")
    ax.set_title("When the chord changes, does the bass always move?")
    ax.set_ylim(0, 1)

    fig.suptitle("Bass-change / chord-change contingency, pooled across all songs, beat-to-beat", fontsize=11)
    fig.tight_layout()
    out = PLOT_ROOT / "bass_chord_change_correlation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")
    print(f"    P(chord change | bass change) = {p_chord_change_given_bass_change:.1%}")
    print(f"    P(chord change | bass same)   = {p_chord_change_given_bass_same:.1%}")
    print(f"    P(bass change | chord change) = {p_bass_change_given_chord_change:.1%}")
    print(f"    P(bass change | chord same)   = {p_bass_change_given_chord_same:.1%}")


# ---------------------------------------------------------------------------
# 3. Run-length comparison
# ---------------------------------------------------------------------------

def compute_bass_run_lengths(all_data: list[dict]) -> list[int]:
    lengths = []
    for d in all_data:
        bass = d["bass"]
        # split into maximal stretches of consecutively-valid beats, run-length
        # encode pitch class within each stretch (a -1 gap breaks continuity --
        # we don't know the bass was really constant across a real silence).
        start = 0
        while start < len(bass):
            if bass[start] < 0:
                start += 1
                continue
            end = start
            while end < len(bass) and bass[end] >= 0:
                end += 1
            pcs = [int(m) % 12 for m in bass[start:end]]
            runs = compress_to_runs(pcs)
            lengths.extend(r.n_beats for r in runs)
            start = end
    return lengths


def plot_run_length_comparison(bass_lengths: list[int], chord_lengths: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    max_len = min(max(bass_lengths + chord_lengths, default=1), 24)
    bins = np.arange(1, max_len + 2) - 0.5
    ax.hist(bass_lengths, bins=bins, alpha=0.6, label=f"bass pitch-class runs (n={len(bass_lengths)}, "
                                                        f"mean={np.mean(bass_lengths):.2f})",
            density=True, color="#1f77b4")
    ax.hist(chord_lengths, bins=bins, alpha=0.6, label=f"GT chord runs (n={len(chord_lengths)}, "
                                                         f"mean={np.mean(chord_lengths):.2f})",
            density=True, color="#d62728")
    ax.set_xlabel("run length (beats)")
    ax.set_ylabel("density")
    ax.set_title("Does the bass change pitch class faster than the chord actually changes?")
    ax.legend()
    fig.tight_layout()
    out = PLOT_ROOT / "run_length_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}  (bass mean={np.mean(bass_lengths):.2f} beats, "
          f"chord mean={np.mean(chord_lengths):.2f} beats)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--songs", nargs="+", default=["001", "002", "003", "004", "005"])
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING)

    PLOT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Collecting bass + GT-chord observations per song...")
    all_data = []
    for song_id in args.songs:
        d = collect_song_data(song_id)
        if d is not None:
            all_data.append(d)
            n_valid_bass = int((d["bass"] >= 0).sum())
            print(f"  {song_id}: {d['B']} beats, {n_valid_bass} with inferred bass, "
                  f"{len(d['chord_run_beats'])} GT chord events")

    print("\nComputing + plotting...")
    counts = compute_scale_degree_hist(all_data)
    plot_scale_degree_hist(counts, len(all_data))

    contingency = compute_change_contingency(all_data)
    plot_change_contingency(contingency)

    bass_lengths = compute_bass_run_lengths(all_data)
    chord_lengths = [n for d in all_data for n in d["chord_run_beats"]]
    plot_run_length_comparison(bass_lengths, chord_lengths)


if __name__ == "__main__":
    main()
