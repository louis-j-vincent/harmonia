"""
Same 2-panel layout (note-probability heatmap + chroma heatmap, sharing a
time axis) as scripts/plot_note_probs_vs_gt.py, with the third panel
replaced by one of two exploratory signals -- see scripts/bass_track.py for
the underlying inference and the rationale (bass motion as a candidate
signal for *when* a chord changes, see docs/known_issues.md #1):

  --panel bass   inferred bass note per beat (lowest confidently-active
                  piano key, from sustain-based note_probs), with GT chord
                  regions shaded faintly behind it for reference.
  --panel key    rolling per-beat key estimate (see
                  bass_track.rolling_key_track), rendered as contiguous
                  coloured runs the same way the GT chord panel renders
                  chords, with the real key_audio.txt ground truth shown as
                  a thin reference strip.

Usage:
    .venv/bin/python scripts/plot_bass_and_key_tracks.py --song 001 --panel bass
    .venv/bin/python scripts/plot_bass_and_key_tracks.py --song 001 --panel key
    .venv/bin/python scripts/plot_bass_and_key_tracks.py --song 001 --panel both
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

from plot_note_probs_vs_gt import MIDI_START, NOTE_NAMES, midi_label, parse_note_name  # noqa: E402
from bass_track import compress_to_runs, forward_fill_bass, infer_bass_track_learned, rolling_key_track  # noqa: E402

DATA_ROOT = Path(__file__).parent.parent / "data"
PLOT_ROOT = Path(__file__).parent.parent / "docs" / "plots"


def _load_song(song_id: str):
    from harmonia.data.pop909_parser import POP909Parser
    from harmonia.models.rhythm import RhythmAnalyser
    from harmonia.models.stage1_pitch import PitchExtractor

    wav = DATA_ROOT / "renders" / "pop909" / song_id / f"{song_id}_v005_musescoregeneral.wav"
    if not wav.exists():
        wav = DATA_ROOT / "renders" / "pop909" / song_id / f"{song_id}_v000_prog0.wav"
    if not wav.exists():
        print(f"No render found for song {song_id}")
        sys.exit(1)

    extractor = PitchExtractor(cache_dir=DATA_ROOT / "cache")
    rhythm = RhythmAnalyser(prefer_madmom=False)
    act = extractor.extract(wav)
    bg = rhythm.analyse(wav)
    beat_probs_onset = bg.quantise_frames(act.frame_times, act.onset_probs)

    gt_song = POP909Parser(DATA_ROOT / "pop909" / "POP909").parse_song(song_id)
    return wav, act, bg, beat_probs_onset, gt_song


def _build_top_panels(fig, gs, beat_probs_onset, low_k, high_k, low_name, high_name, song_id):
    """Panels 0 (zoomed note probs) and 1 (chroma), identical to
    plot_note_probs_vs_gt.py, laid out in a caller-supplied gridspec."""
    B = beat_probs_onset.shape[0]
    zoomed = beat_probs_onset[:, low_k:high_k + 1]

    chroma = np.zeros((B, 12), dtype=np.float64)
    for k in range(88):
        pc = (MIDI_START + k) % 12
        chroma[:, pc] += beat_probs_onset[:, k]
    chroma_norm = chroma / chroma.sum(axis=1, keepdims=True).clip(min=1e-9)

    ax_notes = fig.add_subplot(gs[0, 0])
    ax_chroma = fig.add_subplot(gs[1, 0], sharex=ax_notes)
    cax_notes = fig.add_subplot(gs[0, 1])
    cax_chroma = fig.add_subplot(gs[1, 1])

    beat_max = zoomed.max(axis=1, keepdims=True).clip(min=1e-6)
    display = (zoomed / beat_max).T
    im = ax_notes.imshow(
        display, aspect="auto", origin="lower", cmap="inferno",
        vmin=0, vmax=1, interpolation="nearest", extent=[0, B, 0, display.shape[0]],
    )
    n_keys = high_k - low_k + 1
    c_indices_local = [i for i in range(n_keys) if (MIDI_START + low_k + i) % 12 == 0]
    ax_notes.set_yticks(c_indices_local)
    ax_notes.set_yticklabels([midi_label(low_k + i) for i in c_indices_local], fontsize=8)
    for k in c_indices_local:
        ax_notes.axhline(k, color="white", linewidth=0.3, alpha=0.25)
    ax_notes.set_ylabel("Piano key")
    ax_notes.set_title(f"Note probabilities ({low_name}–{high_name}), chroma — POP909 {song_id}", fontsize=11)
    plt.colorbar(im, cax=cax_notes, label="normalised salience")

    im2 = ax_chroma.imshow(
        chroma_norm.T, aspect="auto", origin="lower", cmap="inferno",
        vmin=0, interpolation="nearest", extent=[0, B, 0, 12],
    )
    ax_chroma.set_yticks(np.arange(12) + 0.5)
    ax_chroma.set_yticklabels(NOTE_NAMES, fontsize=8)
    ax_chroma.set_ylabel("Pitch class")
    plt.colorbar(im2, cax=cax_chroma, label="share of chroma energy")

    return ax_notes, ax_chroma


def _shared_xticks(axes, bg, B):
    step = max(1, B // 40)
    xticks = list(range(0, B, step))
    xticklabels = [f"{bg.beat_times[i]:.1f}s" for i in xticks]
    for ax in axes:
        ax.set_xticks(xticks)
        ax.tick_params(labelbottom=True)
        for x in xticks:
            ax.axvline(x, color="white", linewidth=0.4, alpha=0.15, zorder=0)
    for ax in axes[:-1]:
        ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=6)
    axes[-1].set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=7)


def plot_bass_panel(song_id, low, high, out_suffix=""):
    wav, act, bg, beat_probs_onset, gt_song = _load_song(song_id)
    B = beat_probs_onset.shape[0]
    low_midi, high_midi = parse_note_name(low), parse_note_name(high)
    low_k, high_k = low_midi - MIDI_START, high_midi - MIDI_START

    bass_midi_raw = infer_bass_track_learned(beat_probs_onset)
    bass_midi = forward_fill_bass(bass_midi_raw)
    gt_chords = gt_song.chord_events if gt_song else []

    fig = plt.figure(figsize=(min(B * 0.11 + 2, 34), 12.5), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[4, 2, 1.6, 1], width_ratios=[40, 1], hspace=0.08, wspace=0.015)
    ax_notes, ax_chroma = _build_top_panels(fig, gs, beat_probs_onset, low_k, high_k, low, high, song_id)
    ax_bass = fig.add_subplot(gs[2, 0], sharex=ax_notes)
    ax_chords = fig.add_subplot(gs[3, 0], sharex=ax_notes)

    # Faint GT-chord background for reference, same colour convention as the
    # original chord panel (hue = root pitch class).
    for ev in gt_chords:
        b_start = int(np.searchsorted(bg.beat_times, ev.start_beat, side="left"))
        b_end = int(np.searchsorted(bg.beat_times, ev.end_beat, side="left"))
        b_start, b_end = min(b_start, B), min(max(b_end, b_start + 1), B)
        if b_start >= B:
            continue
        color = plt.get_cmap("hsv")(ev.root / 12.0) if ev.root >= 0 else "#888888"
        ax_bass.axvspan(b_start, b_end, color=color, alpha=0.20, linewidth=0)

    # Bass note track: step line over the forward-filled track (a held note
    # is presumed to keep sounding until a new onset supersedes it, up to
    # forward_fill_bass's max_gap), with markers distinguishing a genuine
    # new onset (solid) from a carried-forward held note (hollow) -- gaps
    # remain blank where neither applies (real silence, or a gap too long
    # to safely assume the note's still held).
    beats = np.arange(B)
    valid = bass_midi >= 0
    is_onset = bass_midi_raw >= 0
    ax_bass.plot(beats[valid], bass_midi[valid], color="black", linewidth=0.8,
                 alpha=0.4, drawstyle="steps-mid")
    ax_bass.scatter(beats[is_onset], bass_midi[is_onset], color="black", s=14,
                     zorder=3, label="onset detected")
    held = valid & ~is_onset
    ax_bass.scatter(beats[held], bass_midi[held], facecolors="none", edgecolors="black",
                     s=14, zorder=3, linewidths=0.6, label="held (forward-filled)")
    ax_bass.legend(loc="upper right", fontsize=7, framealpha=0.8)

    lo_disp = max(int(bass_midi[valid].min()) - 2, MIDI_START) if valid.any() else 24
    hi_disp = min(int(bass_midi[valid].max()) + 2, MIDI_START + 87) if valid.any() else 60
    ax_bass.set_ylim(lo_disp, hi_disp)
    c_ticks = [m for m in range(lo_disp, hi_disp + 1) if m % 12 == 0]
    ax_bass.set_yticks(c_ticks)
    ax_bass.set_yticklabels([f"{NOTE_NAMES[m % 12]}{m // 12 - 1}" for m in c_ticks], fontsize=8)
    ax_bass.set_ylabel("Inferred bass note")
    ax_bass.set_xlim(0, B)

    # GT chord panel, same rendering as plot_note_probs_vs_gt.py's chord
    # track, stacked directly below the bass panel for easy comparison.
    for ev in gt_chords:
        b_start = int(np.searchsorted(bg.beat_times, ev.start_beat, side="left"))
        b_end = int(np.searchsorted(bg.beat_times, ev.end_beat, side="left"))
        b_start, b_end = min(b_start, B), min(max(b_end, b_start + 1), B)
        if b_start >= B:
            continue
        color = plt.get_cmap("hsv")(ev.root / 12.0) if ev.root >= 0 else "#888888"
        ax_chords.add_patch(mpatches.Rectangle(
            (b_start, 0), b_end - b_start, 1,
            facecolor=color, edgecolor="white", linewidth=0.3,
        ))
        ax_chords.text(
            (b_start + b_end) / 2, 0.5, ev.label, ha="center", va="center",
            fontsize=6, rotation=90 if (b_end - b_start) < 3 else 0,
            color="black" if ev.root >= 0 else "white",
        )
    ax_chords.set_xlim(0, B)
    ax_chords.set_ylim(0, 1)
    ax_chords.set_yticks([])
    ax_chords.set_ylabel("GT chord")
    ax_chords.set_xlabel("Time →")

    _shared_xticks([ax_notes, ax_chroma, ax_bass, ax_chords], bg, B)
    fig.suptitle(
        f"POP909 {song_id} — inferred bass note along time "
        "(background shading = GT chord, coloured by root)", fontsize=11,
    )

    out_dir = PLOT_ROOT / "inference" / f"pop909_{song_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"bass_track_{low}_{high}{out_suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_key_panel(song_id, low, high, window=8, out_suffix=""):
    from harmonia.data.pop909_parser import POP909Parser

    wav, act, bg, beat_probs_onset, gt_song = _load_song(song_id)
    B = beat_probs_onset.shape[0]
    low_midi, high_midi = parse_note_name(low), parse_note_name(high)
    low_k, high_k = low_midi - MIDI_START, high_midi - MIDI_START

    track = rolling_key_track(beat_probs_onset, window=window)
    labels = [(p.tonic, p.mode) for p in track]
    extras = [{"key_name": p.key_name, "confidence": p.confidence} for p in track]
    runs = compress_to_runs(labels, extras)

    fig = plt.figure(figsize=(min(B * 0.11 + 2, 34), 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[4, 2, 1.4], width_ratios=[40, 1], hspace=0.08, wspace=0.015)
    ax_notes, ax_chroma = _build_top_panels(fig, gs, beat_probs_onset, low_k, high_k, low, high, song_id)
    ax_key = fig.add_subplot(gs[2, 0], sharex=ax_notes)

    for run in runs:
        tonic, mode = run.label
        color = plt.get_cmap("hsv")(tonic / 12.0)
        conf = run.extra["confidence"]
        ax_key.add_patch(mpatches.Rectangle(
            (run.start_b, 0.15), run.n_beats, 0.85,
            facecolor=color, edgecolor="white", linewidth=0.3,
            alpha=0.35 + 0.65 * conf,  # fainter = less confident
            hatch="///" if mode == "minor" else None,
        ))
        if run.n_beats >= 3:
            ax_key.text(
                (run.start_b + run.end_b) / 2, 0.575, run.extra["key_name"],
                ha="center", va="center", fontsize=6, rotation=0,
            )

    # Real GT key ground truth as a thin reference strip along the bottom.
    gt_song2 = POP909Parser(DATA_ROOT / "pop909" / "POP909").parse_song(song_id)
    if gt_song2 and gt_song2.key_events:
        for ev in gt_song2.key_events:
            b_start = int(np.searchsorted(bg.beat_times, ev.start_s, side="left"))
            b_end = int(np.searchsorted(bg.beat_times, ev.end_s, side="left"))
            b_start, b_end = min(b_start, B), min(max(b_end, b_start + 1), B)
            color = plt.get_cmap("hsv")(ev.tonic / 12.0)
            ax_key.add_patch(mpatches.Rectangle(
                (b_start, 0.0), b_end - b_start, 0.12,
                facecolor=color, edgecolor="none",
                hatch="///" if ev.mode == "minor" else None,
            ))
            ax_key.text(b_start + 1, 0.06, f"GT: {ev.label}", fontsize=7, va="center", color="black")

    ax_key.set_ylim(0, 1)
    ax_key.set_yticks([])
    ax_key.set_ylabel(f"Rolling key\n(±{window} beats)")
    ax_key.set_xlabel("Time →")
    ax_key.set_xlim(0, B)

    _shared_xticks([ax_notes, ax_chroma, ax_key], bg, B)
    fig.suptitle(
        f"POP909 {song_id} — rolling per-beat key estimate "
        "(hatched = minor, opacity = confidence, bottom strip = real GT key)", fontsize=11,
    )

    out_dir = PLOT_ROOT / "inference" / f"pop909_{song_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"key_track_{low}_{high}{out_suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--song", default="001")
    parser.add_argument("--low", default="C2")
    parser.add_argument("--high", default="C5")
    parser.add_argument("--panel", choices=["bass", "key", "both"], default="both")
    parser.add_argument("--window", type=int, default=8, help="rolling key window, +/- beats")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING)

    if args.panel in ("bass", "both"):
        plot_bass_panel(args.song, args.low, args.high)
    if args.panel in ("key", "both"):
        plot_key_panel(args.song, args.low, args.high, window=args.window)


if __name__ == "__main__":
    main()
