"""
Concrete illustrations backing docs/architecture_extensions.md items #9-11,
requested after the first pass was judged too hand-wavy: a real distribution
(not a point estimate) for chord duration, a real explanation (with a plot)
of what "the corpus-wide phase pattern is softer" means, and real computed
examples for the n-gram (#10) and song-form (#11) proposals rather than
just prose.

Produces, under docs/plots/structure_proposal/:
  1. chord_duration_distribution.png -- the FULL empirical PMF over chord
     duration (harmonia/theory/duration_prior.py::fit_duration_prior(),
     already built in session 5 -- this was already a distribution, not a
     median; it just wasn't plotted before), all 909 songs.
  2. phase_variability_across_songs.png -- per-song P(chord change | beat
     phase), shown as a distribution across songs (not just the pooled
     mean) -- this is what "real but softer" concretely means: song 001
     sits at the extreme tail, not at the corpus centre.
  3. ngram_illustration.png + printed table -- real bigram frequencies over
     scale-degree-relative chord transitions, computed from all 909 songs
     via each song's own key_audio.txt (item #10, worked example).
  4. form_clustering_song001.png -- a real (not mocked) prototype run of
     the item #11 section-clustering idea on song 001, using its own
     detected 32-beat period.

Usage:
    .venv/bin/python scripts/plot_structure_proposal_illustrations.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_ROOT = Path(__file__).parent.parent / "data"
PLOT_ROOT = Path(__file__).parent.parent / "docs" / "plots" / "structure_proposal"
POP909_DIR = DATA_ROOT / "pop909" / "POP909"

_INTERVAL_TO_ROMAN = {
    0: "I", 1: "bII", 2: "II", 3: "bIII", 4: "III",
    5: "IV", 6: "bV", 7: "V", 8: "bVI", 9: "VI", 10: "bVII", 11: "VII",
}


# ---------------------------------------------------------------------------
# 1. Chord duration: the FULL distribution, not a point estimate
# ---------------------------------------------------------------------------

def plot_duration_distribution() -> None:
    from harmonia.theory.duration_prior import fit_duration_prior

    prior = fit_duration_prior(POP909_DIR)
    pmf = prior["chord"]  # (32,) -- index d = duration (d+1) beats
    beats = np.arange(1, len(pmf) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.bar(beats[:16], pmf[:16], color="#1f77b4")
    for bar_line in [2, 4, 8, 12, 16]:
        ax.axvline(bar_line, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("chord duration (beats)")
    ax.set_ylabel("P(duration = d)")
    ax.set_title("Full empirical PMF, chord duration in beats\n(all 909 songs, "
                  "dotted lines at 0.5/1/2/3/4-bar marks)")
    for d in [1, 2, 3, 4, 8]:
        if d <= len(pmf):
            ax.text(d, pmf[d - 1] + 0.005, f"{pmf[d-1]:.1%}", ha="center", fontsize=8)

    ax = axes[1]
    cdf = np.cumsum(pmf)
    ax.plot(beats[:16], cdf[:16], marker="o", markersize=3, color="#d62728")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="median")
    ax.set_xlabel("chord duration (beats)")
    ax.set_ylabel("cumulative probability")
    ax.set_title("CDF -- e.g. P(duration <= 2 beats) directly readable")
    ax.legend()
    ax.grid(alpha=0.3)

    mean_d = float(np.sum(beats * pmf))
    mode_d = int(beats[np.argmax(pmf)])
    fig.suptitle(
        f"Chord duration is NOT well-summarized by a median alone: mean={mean_d:.2f} beats, mode={mode_d} beats\n"
        f"P(d=1)={pmf[0]:.1%}, P(d=2)={pmf[1]:.1%}, P(d=4)={pmf[3]:.1%} -- a geometric shape (memoryless) "
        f"would decay monotonically from d=1; this instead peaks at d=2",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = PLOT_ROOT / "chord_duration_distribution.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"  mean={mean_d:.2f} beats, mode={mode_d} beats, median={int(beats[np.searchsorted(cdf, 0.5)])} beats")
    print(f"  P(d=1)={pmf[0]:.1%}  P(d=2)={pmf[1]:.1%}  P(d=3)={pmf[2]:.1%}  P(d=4)={pmf[3]:.1%}  "
          f"P(d<=2)={cdf[1]:.1%}  P(d<=4)={cdf[3]:.1%}")


# ---------------------------------------------------------------------------
# 2. Per-song phase-correlation variability
# ---------------------------------------------------------------------------

def _beat_in_bar_phase(is_downbeat: np.ndarray) -> np.ndarray:
    phase = np.zeros(len(is_downbeat), dtype=int)
    counter, started = 0, False
    for b in range(len(is_downbeat)):
        if is_downbeat[b]:
            counter, started = 0, True
        phase[b] = counter if started else -1
        counter += 1
    return phase


def _load_downbeats(song_id: str) -> np.ndarray | None:
    path = POP909_DIR / song_id / "beat_midi.txt"
    if not path.exists():
        return None
    flags = []
    for line in open(path):
        parts = line.strip().split()
        if len(parts) >= 3:
            flags.append(float(parts[2]) >= 0.5)
    return np.array(flags, dtype=bool)


def analyze_phase_variability(min_transitions: int = 40):
    from harmonia.data.pop909_parser import POP909Parser

    parser = POP909Parser(POP909_DIR)
    songs = parser.parse_all(require_audio=False)

    per_song_rates = []  # list of (song_id, [rate_phase0..3], n_min)
    pooled = {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [0, 0]}

    for song in songs:
        is_downbeat = _load_downbeats(song.song_id)
        if is_downbeat is None or len(is_downbeat) != len(song.beat_times) or not song.chord_events:
            continue
        phase = _beat_in_bar_phase(is_downbeat)
        gt_label = [None] * len(song.beat_times)
        for b, t in enumerate(song.beat_times):
            for ev in song.chord_events:
                if ev.start_beat <= t < ev.end_beat:
                    gt_label[b] = ev.label
                    break

        song_counts = {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [0, 0]}
        n_valid = 0
        for b in range(1, len(song.beat_times)):
            p = phase[b]
            if p not in (0, 1, 2, 3) or gt_label[b] is None or gt_label[b - 1] is None:
                continue
            changed = gt_label[b] != gt_label[b - 1]
            song_counts[p][0] += changed
            song_counts[p][1] += 1
            pooled[p][0] += changed
            pooled[p][1] += 1
            n_valid += 1

        if n_valid >= min_transitions and all(song_counts[p][1] > 0 for p in range(4)):
            rates = [song_counts[p][0] / song_counts[p][1] for p in range(4)]
            per_song_rates.append((song.song_id, rates, n_valid))

    pooled_rates = [pooled[p][0] / max(pooled[p][1], 1) for p in range(4)]
    return per_song_rates, pooled_rates


def plot_phase_variability(per_song_rates, pooled_rates, song_of_interest="001") -> None:
    rates_by_phase = [[r[p] for _, r, _ in per_song_rates] for p in range(4)]
    contrast = [r[0] - r[3] for _, r, _ in per_song_rates]  # downbeat vs last-beat advantage
    song_ids = [sid for sid, _, _ in per_song_rates]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    bp = ax.boxplot(rates_by_phase, positions=range(4), widths=0.5, showfliers=True,
                     patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.5)
    ax.plot(range(4), pooled_rates, "D", color="red", markersize=9, zorder=5,
            label="pooled mean (all songs combined)")
    if song_of_interest in song_ids:
        i = song_ids.index(song_of_interest)
        song_rates = [rates_by_phase[p][i] for p in range(4)]
        ax.plot(range(4), song_rates, "o-", color="#2ca02c", markersize=8, linewidth=2,
                label=f"song {song_of_interest} (this is what we plotted before)")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"beat {p}" for p in range(4)])
    ax.set_ylabel("P(chord changed | this beat phase), per song")
    ax.set_title(f"Each box = distribution ACROSS {len(per_song_rates)} songs\n"
                 "song 001 (green) sits near the extreme, not the centre")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ax.hist(contrast, bins=30, color="#888888")
    if song_of_interest in song_ids:
        i = song_ids.index(song_of_interest)
        ax.axvline(contrast[i], color="#2ca02c", linewidth=2,
                   label=f"song {song_of_interest} = {contrast[i]:.2f}")
    ax.axvline(np.mean(contrast), color="red", linestyle="--",
               label=f"corpus mean = {np.mean(contrast):.2f}")
    ax.set_xlabel("per-song \"downbeat advantage\": P(change|beat 0) - P(change|beat 3)")
    ax.set_ylabel("number of songs")
    ax.set_title("This is what \"real but softer\" means concretely:\n"
                 "the effect is real (mean > 0) but varies a lot song-to-song")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = PLOT_ROOT / "phase_variability_across_songs.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"  {len(per_song_rates)} songs with >=40 valid beat-transitions")
    print(f"  pooled rates by phase: {[f'{r:.1%}' for r in pooled_rates]}")
    print(f"  per-song downbeat-advantage: mean={np.mean(contrast):.2f}, "
          f"std={np.std(contrast):.2f}, min={np.min(contrast):.2f}, max={np.max(contrast):.2f}")
    if song_of_interest in song_ids:
        i = song_ids.index(song_of_interest)
        pct = 100 * (np.array(contrast) < contrast[i]).mean()
        print(f"  song {song_of_interest}'s contrast ({contrast[i]:.2f}) is at the "
              f"{pct:.0f}th percentile of all songs")


# ---------------------------------------------------------------------------
# 3. N-gram illustration (item #10)
# ---------------------------------------------------------------------------

def quality_bucket(quality) -> str:
    from harmonia.theory.chord_vocabulary import get_template
    t = get_template(quality)
    if 3 in t.intervals:
        return "min"
    if 4 in t.intervals:
        return "maj"
    return "other"


def illustrate_ngrams(top_k: int = 15):
    from harmonia.data.pop909_parser import POP909Parser

    parser = POP909Parser(POP909_DIR)
    songs = parser.parse_all(require_audio=False)

    bigrams = Counter()
    degree_only_bigrams = Counter()
    n_songs_used = 0

    for song in songs:
        key_path = POP909_DIR / song.song_id / "key_audio.txt"
        if not key_path.exists() or not song.chord_events:
            continue
        line = open(key_path).readline().split()
        if len(line) < 3:
            continue
        from harmonia.data.pop909_parser import parse_harte_label
        parsed = parse_harte_label(line[2])
        if parsed is None:
            continue
        tonic, _ = parsed
        n_songs_used += 1

        real_events = [ev for ev in song.chord_events if ev.root >= 0]
        for a, b in zip(real_events, real_events[1:]):
            if a.root == b.root and a.quality == b.quality:
                continue  # not a real transition, just re-annotation/held chord split
            deg_a, deg_b = (a.root - tonic) % 12, (b.root - tonic) % 12
            key_a = (_INTERVAL_TO_ROMAN[deg_a], quality_bucket(a.quality))
            key_b = (_INTERVAL_TO_ROMAN[deg_b], quality_bucket(b.quality))
            bigrams[(key_a, key_b)] += 1
            degree_only_bigrams[(_INTERVAL_TO_ROMAN[deg_a], _INTERVAL_TO_ROMAN[deg_b])] += 1

    print(f"\nN-gram illustration: {n_songs_used} songs used, {sum(bigrams.values())} chord-to-chord transitions")
    print(f"\nTop {top_k} (degree, quality)-bucket bigrams:")
    total = sum(bigrams.values())
    top_items = bigrams.most_common(top_k)
    for (ka, kb), c in top_items:
        print(f"  {ka[0]}{ka[1]:<5s} -> {kb[0]}{kb[1]:<5s}   {c:6d}  ({c/total:.2%})")

    print(f"\nTop {top_k} degree-only bigrams (quality collapsed):")
    total2 = sum(degree_only_bigrams.values())
    top_items2 = degree_only_bigrams.most_common(top_k)
    for (da, db), c in top_items2:
        print(f"  {da:<5s} -> {db:<5s}   {c:6d}  ({c/total2:.2%})")

    # Plot: top-15 (degree,quality) bigrams as a horizontal bar chart
    fig, ax = plt.subplots(figsize=(9, 7))
    labels = [f"{ka[0]}{ka[1]} -> {kb[0]}{kb[1]}" for (ka, kb), _ in reversed(top_items)]
    values = [c / total for _, c in reversed(top_items)]
    ax.barh(labels, values, color="#1f77b4")
    for i, v in enumerate(values):
        ax.text(v + 0.0005, i, f"{v:.2%}", va="center", fontsize=8)
    ax.set_xlabel("share of all chord-to-chord transitions (909 songs, pooled)")
    ax.set_title(f"Most common scale-degree-relative chord bigrams\n"
                 f"(item #10 worked example, n={total} transitions)")
    fig.tight_layout()
    out = PLOT_ROOT / "ngram_illustration.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")


# ---------------------------------------------------------------------------
# 4. Form-clustering illustration (item #11), real prototype on song 001
# ---------------------------------------------------------------------------

def illustrate_form_clustering(song_id: str = "001", similarity_threshold: float = 0.85):
    from harmonia.data.pop909_parser import POP909Parser
    from harmonia.models.periodicity import score_periods
    from harmonia.models.rhythm import RhythmAnalyser
    from harmonia.models.stage1_pitch import PitchExtractor
    from harmonia.models.structure import _beat_chroma

    wav = DATA_ROOT / "renders" / "pop909" / song_id / f"{song_id}_v005_musescoregeneral.wav"
    gt_song = POP909Parser(POP909_DIR).parse_song(song_id)
    extractor = PitchExtractor(cache_dir=DATA_ROOT / "cache")
    rhythm = RhythmAnalyser(prefer_madmom=False)
    act = extractor.extract(wav)
    bg = rhythm.analyse(wav)
    beat_probs = bg.quantise_frames(act.frame_times, act.onset_probs)
    B = beat_probs.shape[0]

    periods = score_periods(beat_probs, beats_per_bar=4, top_k=1)
    period = list(periods.keys())[0] if periods else 32
    print(f"\nForm-clustering illustration, song {song_id}: detected period = {period} beats "
          f"(score={periods.get(period, 0):.2f})")

    # Slice into non-overlapping period-length windows.
    n_windows = B // period
    chroma_per_beat = _beat_chroma(beat_probs, norm="l1")  # (B, 12)
    window_chroma = np.array([
        chroma_per_beat[w * period:(w + 1) * period].sum(axis=0) for w in range(n_windows)
    ])
    window_chroma_norm = window_chroma / np.linalg.norm(window_chroma, axis=1, keepdims=True).clip(min=1e-9)

    # Greedy nearest-centroid clustering: assign each window to the most
    # similar existing cluster if above threshold, else start a new one.
    labels = []
    centroids = []  # list of (label_char, mean_vector, count)
    for w in range(n_windows):
        v = window_chroma_norm[w]
        best_label, best_sim = None, -1.0
        for label, centroid, _ in centroids:
            sim = float(np.dot(v, centroid))
            if sim > best_sim:
                best_sim, best_label = sim, label
        if best_sim >= similarity_threshold:
            labels.append(best_label)
            for i, (label, centroid, count) in enumerate(centroids):
                if label == best_label:
                    new_count = count + 1
                    new_centroid = (centroid * count + v) / new_count
                    new_centroid /= np.linalg.norm(new_centroid).clip(min=1e-9)
                    centroids[i] = (label, new_centroid, new_count)
        else:
            new_label = chr(ord("A") + len(centroids))
            labels.append(new_label)
            centroids.append((new_label, v, 1))

    print(f"  {n_windows} windows of {period} beats each -> section labels: {' '.join(labels)}")

    # For a sanity check, print each window's dominant GT chord(s).
    for w in range(n_windows):
        b0, b1 = w * period, (w + 1) * period
        t0, t1 = bg.beat_times[b0], bg.beat_times[min(b1, B - 1)]
        chords_in_window = [ev.label for ev in gt_song.chord_events if ev.start_beat < t1 and ev.end_beat > t0]
        dominant = Counter(chords_in_window).most_common(3)
        print(f"    window {w} [{labels[w]}] {t0:6.1f}s-{t1:6.1f}s: "
              f"{', '.join(c for c, _ in dominant)}")

    # Plot: colour-coded section-label strip + the underlying SSM for context.
    from harmonia.models.structure import build_ssm
    ssm = build_ssm(beat_probs)

    fig, (ax_ssm, ax_labels) = plt.subplots(
        2, 1, figsize=(10, 8.5), gridspec_kw={"height_ratios": [6, 1]}, constrained_layout=True,
    )
    im = ax_ssm.imshow(ssm, cmap="magma", origin="lower", vmin=0, vmax=1)
    for w in range(n_windows + 1):
        ax_ssm.axvline(w * period, color="cyan", linewidth=0.6, alpha=0.6)
        ax_ssm.axhline(w * period, color="cyan", linewidth=0.6, alpha=0.6)
    ax_ssm.set_title(f"Song {song_id}: self-similarity matrix, gridlines at the detected "
                      f"{period}-beat period")
    plt.colorbar(im, ax=ax_ssm, label="cosine similarity", pad=0.01)

    unique_labels = sorted(set(labels))
    color_map = {l: plt.get_cmap("tab10")(i / max(len(unique_labels), 1)) for i, l in enumerate(unique_labels)}
    for w, label in enumerate(labels):
        ax_labels.add_patch(mpatches.Rectangle((w * period, 0), period, 1,
                                                facecolor=color_map[label], edgecolor="white"))
        ax_labels.text(w * period + period / 2, 0.5, label, ha="center", va="center", fontsize=12, weight="bold")
    ax_labels.set_xlim(0, n_windows * period)
    ax_labels.set_ylim(0, 1)
    ax_labels.set_yticks([])
    ax_labels.set_xlabel("beat index")
    ax_labels.set_title(f"Clustered section labels (similarity threshold={similarity_threshold})")

    out = PLOT_ROOT / f"form_clustering_song{song_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main() -> None:
    import logging
    logging.basicConfig(level=logging.WARNING)

    print("=== 1. Chord duration: full distribution ===")
    plot_duration_distribution()

    print("\n=== 2. Per-song phase-correlation variability ===")
    per_song_rates, pooled_rates = analyze_phase_variability()
    plot_phase_variability(per_song_rates, pooled_rates)

    print("\n=== 3. N-gram illustration (item #10) ===")
    illustrate_ngrams()

    print("\n=== 4. Form-clustering illustration (item #11) ===")
    illustrate_form_clustering("002")  # real A/B contrast -- the illustrative case
    illustrate_form_clustering("001")  # degenerates to all-"A" -- the honest null case


if __name__ == "__main__":
    main()
