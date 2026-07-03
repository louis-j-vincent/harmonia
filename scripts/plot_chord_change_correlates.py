"""
Exploratory correlation analysis: which observable per-beat signals actually
predict a real chord change, for one song? Groundwork for the chord-change
detector (docs/known_issues.md #1's unfinished stretch goal) -- run before
committing to any specific detector design.

Uses POP909's OWN ground-truth beat/downbeat grid (beat_midi.txt), not our
audio-derived librosa beat tracker -- for a diagnostic about *timing*, we
want the least noisy possible beat/bar reference. Basic Pitch's onset
activations are then quantised onto that exact grid.

Candidate signals tested against the real chord-change indicator:
  - onset density      (rhythm between notes -- how much attack energy this beat)
  - bass pitch-class change (from the learned bass detector)
  - bass onset present (bass "rhythm" -- is the bass freshly struck this beat)
  - chroma novelty      (local SSM checkerboard novelty, small kernel)
  - beat-in-bar phase   (is this beat a downbeat, from POP909's real markers)

Produces 4 plots under docs/plots/inference/chord_change_correlates/<song>/:
  1. time_aligned.png       -- multi-panel view, downbeats marked, chord
                               changes marked as vertical lines on every panel
  2. beat_in_bar_rate.png   -- P(chord change | beat position in bar)
  3. correlation_leaderboard.png -- point-biserial correlation of each
                               candidate signal vs the chord-change indicator
  4. chord_duration_bars.png -- histogram of chord duration in BARS (not
                               beats) -- the "how many bars per chord" question

Usage:
    .venv/bin/python scripts/plot_chord_change_correlates.py --song 001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from bass_track import forward_fill_bass, infer_bass_track_learned  # noqa: E402
from plot_note_probs_vs_gt import MIDI_START, NOTE_NAMES, midi_label  # noqa: E402

DATA_ROOT = Path(__file__).parent.parent / "data"
PLOT_ROOT = Path(__file__).parent.parent / "docs" / "plots" / "inference" / "chord_change_correlates"


# ---------------------------------------------------------------------------
# POP909's own ground-truth beat/downbeat grid
# ---------------------------------------------------------------------------

def load_pop909_beat_grid(song_id: str):
    """
    beat_midi.txt format: <time_s> <beat_marker> <downbeat_flag>. Column 3 is
    1.0 exactly on downbeats (verified directly: fires every 4th row for
    POP909's 4/4 songs). This is real annotated metrical ground truth, not
    inferred -- used here instead of our librosa beat tracker specifically
    because this analysis is ABOUT timing, so we want the least noisy
    reference available.
    """
    path = DATA_ROOT / "pop909" / "POP909" / song_id / "beat_midi.txt"
    times, is_downbeat = [], []
    for line in open(path):
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        times.append(float(parts[0]))
        is_downbeat.append(float(parts[2]) >= 0.5)
    return np.array(times), np.array(is_downbeat, dtype=bool)


def beat_in_bar_phase(is_downbeat: np.ndarray) -> np.ndarray:
    """Beats since the most recent downbeat (0 = downbeat itself)."""
    phase = np.zeros(len(is_downbeat), dtype=int)
    counter = 0
    started = False
    for b in range(len(is_downbeat)):
        if is_downbeat[b]:
            counter = 0
            started = True
        phase[b] = counter if started else -1
        counter += 1
    return phase


class _PseudoBeatGrid:
    """Minimal shim so we can reuse BeatGrid.quantise_frames() with POP909's
    own beat times instead of building a real BeatGrid (which wants a
    time_signature enum, backend string, etc. we don't need here)."""
    def __init__(self, beat_times):
        self.beat_times = beat_times

    @property
    def beat_duration_s(self):
        return float(np.median(np.diff(self.beat_times))) if len(self.beat_times) > 1 else 0.5

    def quantise_frames(self, frame_times, note_probs):
        from harmonia.models.rhythm import BeatGrid
        return BeatGrid.quantise_frames(self, frame_times, note_probs)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect(song_id: str) -> dict:
    from harmonia.data.pop909_parser import POP909Parser
    from harmonia.models.stage1_pitch import PitchExtractor
    from harmonia.models.structure import build_ssm, compute_novelty

    wav = DATA_ROOT / "renders" / "pop909" / song_id / f"{song_id}_v005_musescoregeneral.wav"
    gt_song = POP909Parser(DATA_ROOT / "pop909" / "POP909").parse_song(song_id)

    beat_times, is_downbeat = load_pop909_beat_grid(song_id)
    grid = _PseudoBeatGrid(beat_times)
    B = len(beat_times)

    extractor = PitchExtractor(cache_dir=DATA_ROOT / "cache")
    act = extractor.extract(wav)
    beat_probs = grid.quantise_frames(act.frame_times, act.onset_probs)

    # GT chord + change indicator, aligned to this beat grid.
    gt_root = np.full(B, -2, dtype=int)
    gt_label = [None] * B
    for b in range(B):
        t = beat_times[b]
        found = None
        for ev in gt_song.chord_events:
            if ev.start_beat <= t < ev.end_beat:
                found = ev
                break
        if found is not None:
            gt_root[b] = found.root
            gt_label[b] = found.label
    chord_changed = np.zeros(B, dtype=bool)
    for b in range(1, B):
        if gt_root[b] >= 0 and gt_root[b - 1] >= 0:
            chord_changed[b] = gt_label[b] != gt_label[b - 1]

    # Candidate signals
    onset_density = beat_probs.sum(axis=1)

    bass_raw = infer_bass_track_learned(beat_probs)
    bass_filled = forward_fill_bass(bass_raw)
    bass_onset = bass_raw >= 0
    bass_changed = np.zeros(B, dtype=bool)
    for b in range(1, B):
        if bass_filled[b] >= 0 and bass_filled[b - 1] >= 0:
            bass_changed[b] = (bass_filled[b] % 12) != (bass_filled[b - 1] % 12)

    ssm = build_ssm(beat_probs)
    novelty = compute_novelty(ssm, kernel_size=2)  # small kernel: chord-scale, not section-scale
    novelty_full = np.zeros(B)
    novelty_full[:len(novelty)] = novelty

    phase = beat_in_bar_phase(is_downbeat)

    return {
        "song_id": song_id, "B": B, "beat_times": beat_times, "is_downbeat": is_downbeat,
        "phase": phase, "beat_probs": beat_probs, "gt_root": gt_root, "gt_label": gt_label,
        "chord_changed": chord_changed, "onset_density": onset_density,
        "bass_filled": bass_filled, "bass_onset": bass_onset, "bass_changed": bass_changed,
        "novelty": novelty_full, "gt_chords": gt_song.chord_events,
    }


# ---------------------------------------------------------------------------
# Plot 1: time-aligned multi-panel view
# ---------------------------------------------------------------------------

def plot_time_aligned(d: dict, low="C2", high="C5") -> None:
    from plot_note_probs_vs_gt import parse_note_name

    B = d["B"]
    low_k, high_k = parse_note_name(low) - MIDI_START, parse_note_name(high) - MIDI_START
    zoomed = d["beat_probs"][:, low_k:high_k + 1]
    beat_max = zoomed.max(axis=1, keepdims=True).clip(min=1e-6)
    display = (zoomed / beat_max).T

    fig = plt.figure(figsize=(min(B * 0.11 + 2, 34), 13), constrained_layout=True)
    gs = fig.add_gridspec(5, 1, height_ratios=[3, 1, 1, 1, 1], hspace=0.15)
    ax_notes = fig.add_subplot(gs[0, 0])
    ax_chords = fig.add_subplot(gs[1, 0], sharex=ax_notes)
    ax_onset = fig.add_subplot(gs[2, 0], sharex=ax_notes)
    ax_bass = fig.add_subplot(gs[3, 0], sharex=ax_notes)
    ax_novelty = fig.add_subplot(gs[4, 0], sharex=ax_notes)
    axes = [ax_notes, ax_chords, ax_onset, ax_bass, ax_novelty]

    ax_notes.imshow(display, aspect="auto", origin="lower", cmap="inferno",
                     vmin=0, vmax=1, interpolation="nearest", extent=[0, B, 0, display.shape[0]])
    n_keys = high_k - low_k + 1
    c_idx = [i for i in range(n_keys) if (MIDI_START + low_k + i) % 12 == 0]
    ax_notes.set_yticks(c_idx)
    ax_notes.set_yticklabels([midi_label(low_k + i) for i in c_idx], fontsize=8)
    ax_notes.set_ylabel("Piano key")
    ax_notes.set_title(f"POP909 {d['song_id']} — chord-change correlates (downbeats = thick white lines, "
                        "chord changes = red dashed lines)", fontsize=11)

    for ev in d["gt_chords"]:
        b_start = int(np.searchsorted(d["beat_times"], ev.start_beat, side="left"))
        b_end = int(np.searchsorted(d["beat_times"], ev.end_beat, side="left"))
        b_start, b_end = min(b_start, B), min(max(b_end, b_start + 1), B)
        if b_start >= B or ev.root < 0 and ev.label != "N":
            pass
        color = plt.get_cmap("hsv")(ev.root / 12.0) if ev.root >= 0 else "#888888"
        ax_chords.add_patch(mpatches.Rectangle((b_start, 0), b_end - b_start, 1,
                                                facecolor=color, edgecolor="white", linewidth=0.3))
        if b_end - b_start >= 2:
            ax_chords.text((b_start + b_end) / 2, 0.5, ev.label, ha="center", va="center", fontsize=6)
    ax_chords.set_xlim(0, B)
    ax_chords.set_ylim(0, 1)
    ax_chords.set_yticks([])
    ax_chords.set_ylabel("GT chord")

    ax_onset.bar(np.arange(B), d["onset_density"], color="#1f77b4", width=0.8)
    ax_onset.set_ylabel("onset\ndensity")

    ax_bass.plot(np.arange(B), d["bass_filled"], color="black", linewidth=0.7, alpha=0.5, drawstyle="steps-mid")
    onset_mask = d["bass_onset"]
    ax_bass.scatter(np.nonzero(onset_mask)[0], d["bass_filled"][onset_mask], color="black", s=10, zorder=3)
    ax_bass.set_ylabel("bass note")
    valid = d["bass_filled"] >= 0
    if valid.any():
        ax_bass.set_ylim(int(d["bass_filled"][valid].min()) - 2, int(d["bass_filled"][valid].max()) + 2)

    ax_novelty.plot(np.arange(B), d["novelty"], color="#d62728", linewidth=1.0)
    ax_novelty.set_ylabel("chroma\nnovelty")
    ax_novelty.set_xlabel("beat index (POP909 ground-truth grid)")

    for ax in axes:
        for b in np.nonzero(d["is_downbeat"])[0]:
            ax.axvline(b, color="white" if ax is ax_notes else "#aaaaaa", linewidth=1.2, alpha=0.6, zorder=0)
        for b in np.nonzero(d["chord_changed"])[0]:
            ax.axvline(b, color="red", linewidth=0.8, alpha=0.55, linestyle="--", zorder=1)
        ax.set_xlim(0, B)

    out_dir = PLOT_ROOT / d["song_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "time_aligned.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2: P(chord change | beat-in-bar phase)
# ---------------------------------------------------------------------------

def plot_beat_in_bar_rate(d: dict) -> None:
    phase, changed = d["phase"], d["chord_changed"]
    valid = phase >= 0
    max_phase = phase[valid].max() if valid.any() else 3
    rates, counts = [], []
    for p in range(max_phase + 1):
        mask = valid & (phase == p)
        counts.append(int(mask.sum()))
        rates.append(float(changed[mask].mean()) if mask.any() else 0.0)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(range(max_phase + 1), rates, color=["#2ca02c"] + ["#1f77b4"] * max_phase)
    for i, (r, n) in enumerate(zip(rates, counts)):
        ax.text(i, r + 0.01, f"{r:.1%}\n(n={n})", ha="center", fontsize=8)
    ax.set_xticks(range(max_phase + 1))
    ax.set_xticklabels([f"beat {p}" + (" (downbeat)" if p == 0 else "") for p in range(max_phase + 1)])
    ax.set_ylabel("P(chord changed this beat)")
    ax.set_title(f"POP909 {d['song_id']} — does chord change cluster on the downbeat?")
    fig.tight_layout()
    out_dir = PLOT_ROOT / d["song_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "beat_in_bar_rate.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3: correlation leaderboard
# ---------------------------------------------------------------------------

def plot_correlation_leaderboard(d: dict) -> None:
    changed = d["chord_changed"].astype(float)
    is_downbeat = d["is_downbeat"].astype(float)

    signals = {
        "onset density": d["onset_density"],
        "bass pitch-class\nchanged": d["bass_changed"].astype(float),
        "bass onset\npresent (rhythm)": d["bass_onset"].astype(float),
        "chroma novelty\n(local)": d["novelty"],
        "is downbeat": is_downbeat,
    }
    corrs = {}
    for name, sig in signals.items():
        if np.std(sig) == 0 or np.std(changed) == 0:
            corrs[name] = 0.0
        else:
            corrs[name] = float(np.corrcoef(sig, changed)[0, 1])

    items = sorted(corrs.items(), key=lambda kv: kv[1], reverse=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for _, v in items]
    ax.barh([k for k, _ in items], [v for _, v in items], color=colors)
    for i, (_, v) in enumerate(items):
        ax.text(v + (0.005 if v >= 0 else -0.005), i, f"{v:.2f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("point-biserial correlation with real chord-change indicator")
    ax.set_title(f"POP909 {d['song_id']} — which per-beat signal best predicts a chord change?")
    fig.tight_layout()
    out_dir = PLOT_ROOT / d["song_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "correlation_leaderboard.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")
    for k, v in items:
        print(f"    {k.replace(chr(10), ' '):<30s} r={v:+.3f}")


# ---------------------------------------------------------------------------
# Plot 4: chord duration in bars
# ---------------------------------------------------------------------------

def plot_chord_duration_bars(d: dict, beats_per_bar: int = 4) -> None:
    durations_beats = []
    cur_len = 0
    for b in range(d["B"]):
        if d["gt_root"][b] < 0:
            if cur_len:
                durations_beats.append(cur_len)
            cur_len = 0
            continue
        if b > 0 and d["chord_changed"][b]:
            durations_beats.append(cur_len)
            cur_len = 0
        cur_len += 1
    if cur_len:
        durations_beats.append(cur_len)
    durations_bars = [n / beats_per_bar for n in durations_beats]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(durations_bars, bins=np.arange(0, max(durations_bars) + 0.5, 0.25), color="#1f77b4")
    ax.axvline(np.median(durations_bars), color="red", linestyle="--",
               label=f"median = {np.median(durations_bars):.2f} bars")
    ax.set_xlabel("chord duration (bars)")
    ax.set_ylabel("count")
    ax.set_title(f"POP909 {d['song_id']} — how many bars does a chord typically last?")
    ax.legend()
    fig.tight_layout()
    out_dir = PLOT_ROOT / d["song_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "chord_duration_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--song", default="001")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING)

    print(f"Collecting data for song {args.song}...")
    d = collect(args.song)
    print(f"  {d['B']} beats, {d['is_downbeat'].sum()} downbeats, "
          f"{d['chord_changed'].sum()} chord changes")

    plot_time_aligned(d)
    plot_beat_in_bar_rate(d)
    plot_correlation_leaderboard(d)
    plot_chord_duration_bars(d)


if __name__ == "__main__":
    main()
