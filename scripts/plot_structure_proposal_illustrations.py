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


# ---------------------------------------------------------------------------
# Precise diatonic-membership checking now lives in scale_taxonomy.py --
# atomic scale families (major-family covers all 7 "church modes" at once;
# harmonic-minor-family added for completeness), each defined once and
# re-indexed per transposition rather than hand-maintaining a separate
# table per mode. See that module's docstring for the reasoning, and
# docs/known_issues.md / architecture_extensions.md for what changed
# (short version: the old 2-scale-only version here was mathematically
# equivalent to the major-family/parallel-borrow part of the new one --
# verified directly, see scale_taxonomy.py's self-test -- so no numbers
# from before this point are invalidated; harmonic-minor-family only adds
# genuinely new coverage for the rare augmented-mediant chord, since the
# common "harmonic minor V/V7/vii°" chords turn out to be pitch-identical
# to the parallel major's own V/V7/vii° and were already being classified
# correctly as parallel_borrow).
# ---------------------------------------------------------------------------

from scale_taxonomy import (  # noqa: E402
    classify_membership, precise_triad_quality, MAJOR_FAMILY, DIATONIC_MAJOR_FAMILY,
)


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


def illustrate_ngrams_canonical(top_k: int = 15):
    """
    Same idea as illustrate_ngrams(), but fixes a real fragmentation bug
    identified 2026-07-03: that function computed each chord's scale degree
    relative to the song's OWN annotated tonic regardless of mode, so a
    minor-key song's "v -> i" cadence (intervals 7 -> 0) and a major-key
    song's "iii -> vi" motion (intervals 4 -> 9) get tallied as different
    bigrams, even though they're the same relative-pitch event: a minor key
    and its relative major share the exact same 7 notes (verified directly:
    A minor's diatonic set == C major's diatonic set == {0,2,3,5,7,8,10} from
    A / {0,2,4,5,7,9,11} from C, same absolute pitch classes either way).

    Fix: canonicalise every song's reference tonic to its RELATIVE MAJOR
    tonic (tonic+3 semitones for songs annotated as minor; unchanged for
    songs annotated as major) before computing scale degrees. Bigrams are
    tallied into two separate Counters by the song's ORIGINAL annotated
    mode, so we can check directly whether canonicalising actually unifies
    the harmonic language (does minor-key "v->i", now labelled "III->VI",
    show up with similar prominence to major-key "iii->vi"?) or whether
    major- and minor-key songs still look meaningfully different even in a
    shared frame.

    Note: this only undoes the relative-major/minor fragmentation. It does
    NOT address the separate parallel-minor/modal-mixture case (bVI, bVII,
    bIII appearing as literally the same-tonic natural-minor degrees) --
    that would need a same-tonic mode/colour flag, not a tonic shift, and
    is a distinct follow-up.
    """
    from harmonia.data.pop909_parser import POP909Parser, parse_harte_label

    parser = POP909Parser(POP909_DIR)
    songs = parser.parse_all(require_audio=False)

    bigrams_by_mode = {"major": Counter(), "minor": Counter()}
    n_songs_by_mode = {"major": 0, "minor": 0}

    for song in songs:
        key_path = POP909_DIR / song.song_id / "key_audio.txt"
        if not key_path.exists() or not song.chord_events:
            continue
        line = open(key_path).readline().split()
        if len(line) < 3:
            continue
        parsed = parse_harte_label(line[2])
        if parsed is None:
            continue
        tonic, key_quality = parsed
        mode = "major" if quality_bucket(key_quality) == "maj" else "minor"
        canonical_tonic = tonic if mode == "major" else (tonic + 3) % 12
        n_songs_by_mode[mode] += 1

        real_events = [ev for ev in song.chord_events if ev.root >= 0]
        for a, b in zip(real_events, real_events[1:]):
            if a.root == b.root and a.quality == b.quality:
                continue
            deg_a = (a.root - canonical_tonic) % 12
            deg_b = (b.root - canonical_tonic) % 12
            key_a = (_INTERVAL_TO_ROMAN[deg_a], quality_bucket(a.quality))
            key_b = (_INTERVAL_TO_ROMAN[deg_b], quality_bucket(b.quality))
            bigrams_by_mode[mode][(key_a, key_b)] += 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    highlighted_bigrams = {(("V", "maj"), ("I", "maj")), (("III", "min"), ("VI", "min"))}

    for ax, mode in zip(axes, ["major", "minor"]):
        bigrams = bigrams_by_mode[mode]
        total = sum(bigrams.values())
        top_items = bigrams.most_common(top_k)
        print(f"\nTop {top_k} bigrams, {mode}-annotated songs "
              f"(n_songs={n_songs_by_mode[mode]}, n_transitions={total}, "
              f"canonical tonic = {'song tonic' if mode == 'major' else 'song tonic + 3 (relative major)'}):")
        for (ka, kb), c in top_items:
            print(f"  {ka[0]}{ka[1]:<5s} -> {kb[0]}{kb[1]:<5s}   {c:6d}  ({c/total:.2%})")

        labels = [f"{ka[0]}{ka[1]} -> {kb[0]}{kb[1]}" for (ka, kb), _ in reversed(top_items)]
        values = [c / total for _, c in reversed(top_items)]
        colors = ["#d62728" if (ka, kb) in highlighted_bigrams else "#1f77b4"
                  for (ka, kb), _ in reversed(top_items)]
        ax.barh(labels, values, color=colors)
        for i, v in enumerate(values):
            ax.text(v + 0.0005, i, f"{v:.2%}", va="center", fontsize=8)
        ax.set_xlabel("share of this mode's chord-to-chord transitions")
        tonic_desc = "canonical tonic = song's own major tonic" if mode == "major" \
            else "canonical tonic = relative major (song tonic + 3)"
        ax.set_title(f"{mode.upper()}-annotated songs (n={total} transitions)\n{tonic_desc}")

    fig.suptitle(
        "Same canonical (relative-major) reference frame for both groups -- red bars mark "
        "V->I (major) and its canonicalised minor-key equivalent III->VI (minor)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = PLOT_ROOT / "ngram_illustration_canonical_major_vs_minor.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")

    # Direct check requested: does minor-key "v->i" (canonicalised to III->VI)
    # show comparable prominence to major-key iii->vi, and does major-key V->I
    # show up at all in the minor-annotated group (it shouldn't, structurally)?
    maj_total = sum(bigrams_by_mode["major"].values())
    min_total = sum(bigrams_by_mode["minor"].values())
    v_i_major = bigrams_by_mode["major"][(("V", "maj"), ("I", "maj"))]
    iii_vi_major = bigrams_by_mode["major"][(("III", "min"), ("VI", "min"))]
    v_i_minor = bigrams_by_mode["minor"][(("V", "maj"), ("I", "maj"))]
    iii_vi_minor = bigrams_by_mode["minor"][(("III", "min"), ("VI", "min"))]
    print("\nDirect comparison:")
    print(f"  V(maj)->I(maj)   in MAJOR songs: {v_i_major/maj_total:.2%}   in MINOR songs: {v_i_minor/min_total:.2%}")
    print(f"  III(min)->VI(min) in MAJOR songs: {iii_vi_major/maj_total:.2%}   in MINOR songs: {iii_vi_minor/min_total:.2%}")


# ---------------------------------------------------------------------------
# 3b. Fold the diatonic/parallel-borrow/chromatic taxonomy into the
#     canonical (relative-major) bigram work, with plots.
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "diatonic": "#1f77b4", "parallel_borrow": "#2ca02c", "harmonic_minor_borrow": "#ff7f0e",
    "secondary_dominant": "#d62728", "sus": "#888888", "other_chromatic": "#9467bd",
}
_TQ_TO_KEY_MODE = {"maj": "major", "min": "minor"}


def illustrate_taxonomy_overview():
    """Plot 1 of the fold-in: proportion of each membership category, per
    song-annotated-mode, using each song's OWN tonic (parallel comparison --
    this is the categorisation, independent of the canonical relabelling
    used for bigram pooling below)."""
    from harmonia.data.pop909_parser import POP909Parser, parse_harte_label

    parser = POP909Parser(POP909_DIR)
    songs = parser.parse_all(require_audio=False)

    counts = {"major": Counter(), "minor": Counter()}
    for song in songs:
        key_path = POP909_DIR / song.song_id / "key_audio.txt"
        if not key_path.exists() or not song.chord_events:
            continue
        line = open(key_path).readline().split()
        if len(line) < 3:
            continue
        parsed = parse_harte_label(line[2])
        if parsed is None:
            continue
        tonic, key_quality = parsed
        mode = _TQ_TO_KEY_MODE.get(quality_bucket(key_quality), "major")
        for ev in song.chord_events:
            if ev.root < 0:
                continue
            interval = (ev.root - tonic) % 12
            cat = classify_membership(interval, ev.quality, song_mode=mode)
            counts[mode][cat] += 1

    # classify_membership() already resolves "own" vs "parallel-other"
    # relative to the song's own annotated mode, so no per-group category
    # swap is needed here (unlike the first pass at this plot).
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cat_labels = ["diatonic to\nown mode", "parallel-mode\nborrow", "harmonic-minor\nborrow (rare)", "sus\n(neutral)", "chromatic\n(neither)"]
    cats = ["diatonic_own", "parallel_borrow", "harmonic_minor_borrow", "sus", "chromatic"]
    x = np.arange(len(cat_labels))
    width = 0.35
    for i, mode in enumerate(["major", "minor"]):
        total = sum(counts[mode].values())
        vals = [counts[mode][c] / total for c in cats]
        ax.bar(x + (i - 0.5) * width, vals, width, label=f"{mode}-annotated (n={total})")
        for xi, v in zip(x + (i - 0.5) * width, vals):
            ax.text(xi, v + 0.01, f"{v:.1%}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels)
    ax.set_ylabel("share of chord events")
    ax.set_title("Diatonic-membership breakdown, checked against BOTH scales\nat each song's own tonic (parallel major/minor comparison)")
    ax.legend()
    fig.tight_layout()
    out = PLOT_ROOT / "taxonomy_overview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def _collect_categorized_bigrams():
    """Shared data-collection pass for illustrate_ngrams_by_category() and
    illustrate_chromatic_only_bigrams(): canonical (relative-major-pooled)
    bigram counts, plus each bigram's dominant membership-category tag
    (checked against each song's OWN tonic/mode, not the canonical one --
    category is a property of the real felt tonal centre; canonical
    relabelling is only used to POOL major- and minor-song-equivalent
    motions under the same text label)."""
    from harmonia.data.pop909_parser import POP909Parser, parse_harte_label

    parser = POP909Parser(POP909_DIR)
    songs = parser.parse_all(require_audio=False)

    bigrams_by_mode = {"major": Counter(), "minor": Counter()}
    bigram_category = {"major": {}, "minor": {}}

    for song in songs:
        key_path = POP909_DIR / song.song_id / "key_audio.txt"
        if not key_path.exists() or not song.chord_events:
            continue
        line = open(key_path).readline().split()
        if len(line) < 3:
            continue
        parsed = parse_harte_label(line[2])
        if parsed is None:
            continue
        tonic, key_quality = parsed
        mode = _TQ_TO_KEY_MODE.get(quality_bucket(key_quality), "major")
        canonical_tonic = tonic if mode == "major" else (tonic + 3) % 12

        real_events = [ev for ev in song.chord_events if ev.root >= 0]
        for a, b in zip(real_events, real_events[1:]):
            if a.root == b.root and a.quality == b.quality:
                continue
            # canonical labels, for pooling major/minor-equivalent motions
            deg_a = (a.root - canonical_tonic) % 12
            deg_b = (b.root - canonical_tonic) % 12
            key_a = (_INTERVAL_TO_ROMAN[deg_a], quality_bucket(a.quality))
            key_b = (_INTERVAL_TO_ROMAN[deg_b], quality_bucket(b.quality))
            bigram_key = (key_a, key_b)
            bigrams_by_mode[mode][bigram_key] += 1

            # category, using the song's OWN (un-shifted) tonic -- a property
            # of the real tonal centre, not the canonical relabelling.
            interval_a_own = (a.root - tonic) % 12
            interval_b_own = (b.root - tonic) % 12
            cat_a = classify_membership(interval_a_own, a.quality, song_mode=mode)
            cat_b = classify_membership(interval_b_own, b.quality, song_mode=mode)
            # tag a bigram by its most exotic member.
            _RANK = {"chromatic": 0, "harmonic_minor_borrow": 1, "parallel_borrow": 2,
                     "sus": 3, "diatonic_own": 4}
            tag_rank = min(_RANK[cat_a], _RANK[cat_b])
            tag = {0: "secondary_dominant", 1: "harmonic_minor_borrow", 2: "parallel_borrow",
                   3: "sus", 4: "diatonic"}[tag_rank]
            bigram_category[mode].setdefault(bigram_key, Counter())[tag] += 1

    return bigrams_by_mode, bigram_category


def illustrate_ngrams_by_category(top_k: int = 15):
    """Plot 2 of the fold-in: the canonical (relative-major-pooled) bigram
    tables from illustrate_ngrams_canonical(), now with each bar coloured by
    the membership category of its FIRST chord."""
    bigrams_by_mode, bigram_category = _collect_categorized_bigrams()

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    for ax, mode in zip(axes, ["major", "minor"]):
        bigrams = bigrams_by_mode[mode]
        total = sum(bigrams.values())
        top_items = bigrams.most_common(top_k)
        labels = [f"{ka[0]}{ka[1]} -> {kb[0]}{kb[1]}" for (ka, kb), _ in reversed(top_items)]
        values = [c / total for _, c in reversed(top_items)]
        colors = []
        for (bigram_key, _) in reversed(top_items):
            dominant_tag = bigram_category[mode][bigram_key].most_common(1)[0][0]
            colors.append(CATEGORY_COLORS.get(dominant_tag, "#888888"))
        ax.barh(labels, values, color=colors)
        for i, v in enumerate(values):
            ax.text(v + 0.0005, i, f"{v:.2%}", va="center", fontsize=8)
        ax.set_xlabel("share of this mode's chord-to-chord transitions")
        ax.set_title(f"{mode.upper()}-annotated songs (n={total} transitions)")

    handles = [mpatches.Patch(color=c, label=l) for l, c in
               [("diatonic (both chords in-scale)", CATEGORY_COLORS["diatonic"]),
                ("parallel-mode borrow (e.g. bVI/bVII, iv)", CATEGORY_COLORS["parallel_borrow"]),
                ("harmonic-minor-only borrow (rare: aug mediant)", CATEGORY_COLORS["harmonic_minor_borrow"]),
                ("chromatic (secondary dominant, etc.)", CATEGORY_COLORS["secondary_dominant"]),
                ("involves a sus chord", CATEGORY_COLORS["sus"])]]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Same canonical bigrams as before, now coloured by the more exotic chord's "
                 "diatonic-membership category", fontsize=11)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    out = PLOT_ROOT / "ngram_by_category.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def illustrate_chromatic_only_bigrams(top_k: int = 15):
    """
    The top-15-by-raw-frequency view above is dominated by plain diatonic
    motion (chromatic/borrowed chords are individually too rare to crack a
    frequency-sorted top-15, even though they're collectively ~7-11% of all
    chords -- see taxonomy_overview.png). This filters to bigrams whose
    dominant tag is NOT "diatonic", so the non-diatonic vocabulary is
    actually visible instead of being drowned out.
    """
    bigrams_by_mode, bigram_category = _collect_categorized_bigrams()

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    for ax, mode in zip(axes, ["major", "minor"]):
        bigrams = bigrams_by_mode[mode]
        total_all = sum(bigrams.values())
        non_diatonic = Counter({
            k: c for k, c in bigrams.items()
            if bigram_category[mode][k].most_common(1)[0][0] != "diatonic"
        })
        total_non_diatonic = sum(non_diatonic.values())
        top_items = non_diatonic.most_common(top_k)

        labels = [f"{ka[0]}{ka[1]} -> {kb[0]}{kb[1]}" for (ka, kb), _ in reversed(top_items)]
        values = [c / total_all for _, c in reversed(top_items)]  # share of ALL transitions, for comparability
        colors = []
        for (bigram_key, _) in reversed(top_items):
            dominant_tag = bigram_category[mode][bigram_key].most_common(1)[0][0]
            colors.append(CATEGORY_COLORS.get(dominant_tag, "#888888"))
        ax.barh(labels, values, color=colors)
        for i, v in enumerate(values):
            ax.text(v + 0.0002, i, f"{v:.2%}", va="center", fontsize=8)
        ax.set_xlabel("share of ALL this mode's transitions (not just non-diatonic ones)")
        ax.set_title(f"{mode.upper()}-annotated songs\n"
                     f"non-diatonic: {total_non_diatonic}/{total_all} transitions ({total_non_diatonic/total_all:.1%})")

    handles = [mpatches.Patch(color=c, label=l) for l, c in
               [("parallel-mode borrow (e.g. bVI/bVII, iv)", CATEGORY_COLORS["parallel_borrow"]),
                ("harmonic-minor-only borrow (rare: aug mediant)", CATEGORY_COLORS["harmonic_minor_borrow"]),
                ("chromatic (secondary dominant, etc.)", CATEGORY_COLORS["secondary_dominant"]),
                ("involves a sus chord", CATEGORY_COLORS["sus"])]]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Top NON-diatonic bigrams only -- what the previous plot's frequency-sorted "
                 "top-15 was hiding", fontsize=11)
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    out = PLOT_ROOT / "ngram_chromatic_only.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 3c. Are 7ths the differentiator? Empirical resolution-rate test.
# ---------------------------------------------------------------------------

def illustrate_seventh_differentiation():
    """
    Tests directly (not by music-theory assumption): among chords that are
    triad-quality "maj" (i.e. would all be lumped together by a coarse
    maj/min bucket), does the SPECIFIC raw quality -- plain major triad vs
    maj7 vs dom7 -- predict how reliably the chord resolves down a perfect
    fifth to the next chord (the functional-dominant signature)? If dom7
    behaves very differently from bare maj/maj7 even at the SAME chromatic
    scale position, the 7th is doing real, simplifying work: "is this a
    dom7" becomes a much more direct test for "is this an applied dominant"
    than "is this a chromatic major triad at a plausible secondary-dominant
    scale degree."
    """
    from harmonia.data.pop909_parser import POP909Parser, parse_harte_label

    parser = POP909Parser(POP909_DIR)
    songs = parser.parse_all(require_audio=False)

    resolve = Counter()
    total = Counter()

    for song in songs:
        key_path = POP909_DIR / song.song_id / "key_audio.txt"
        if not key_path.exists() or not song.chord_events:
            continue
        line = open(key_path).readline().split()
        if len(line) < 3:
            continue
        parsed = parse_harte_label(line[2])
        if parsed is None:
            continue
        tonic, key_quality = parsed
        mode = _TQ_TO_KEY_MODE.get(quality_bucket(key_quality), "major")

        real_events = [ev for ev in song.chord_events if ev.root >= 0]
        for a, b in zip(real_events, real_events[1:]):
            if a.root == b.root and a.quality == b.quality:
                continue
            interval_a = (a.root - tonic) % 12
            cat = classify_membership(interval_a, a.quality, song_mode=mode)
            tq = precise_triad_quality(a.quality)
            resolves = int(b.root == (a.root - 7) % 12)

            if tq == "maj":
                if cat == "chromatic":
                    group = f"chromatic {a.quality.value}"
                elif interval_a == 7:
                    group = f"diatonic V {a.quality.value}"
                else:
                    group = None
                if group:
                    resolve[group] += resolves
                    total[group] += 1
            resolve["ANY chord (baseline)"] += resolves
            total["ANY chord (baseline)"] += 1

    groups = [g for g in total if total[g] >= 30]
    groups.sort(key=lambda g: -resolve[g] / total[g])
    rates = [resolve[g] / total[g] for g in groups]
    ns = [total[g] for g in groups]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#d62728" if "dom" in g.lower() or g.endswith(" 7") else
              "#2ca02c" if "maj7" in g else
              "#1f77b4" if "maj " in g and "7" not in g else "#888888" for g in groups]
    bars = ax.barh(groups, rates, color=colors)
    for i, (r, n) in enumerate(zip(rates, ns)):
        ax.text(r + 0.01, i, f"{r:.1%} (n={n})", va="center", fontsize=8)
    ax.set_xlabel("P(next chord's root is a perfect 5th below this chord's root)")
    ax.set_title("Does the SPECIFIC 7th type predict functional-dominant behaviour,\n"
                 "beyond just \"major triad in a chromatic position\"?")
    ax.set_xlim(0, max(rates) + 0.15)
    fig.tight_layout()
    out = PLOT_ROOT / "seventh_differentiation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    for g, r, n in zip(groups, rates, ns):
        print(f"  {g:<28s} n={n:6d}  resolve-down-5th={r:.1%}")


# ---------------------------------------------------------------------------
# 3d. Mode-agnostic parent-scale identification: the simpler 12-way
#     "which collection" problem, decoupled from the harder "which degree
#     is home" problem (see scale_taxonomy.py's module docstring).
# ---------------------------------------------------------------------------

def identify_best_parent_scale(chord_events) -> tuple[int, float]:
    """
    For a song's real chord events, find the major-family transposition
    (Ionian-reference tonic T, 0-11) whose diatonic-triad table matches the
    most chord (root, quality) pairs -- exact triad match, not just root
    pitch-class membership. Returns (best_T, fraction_matched).

    This is deliberately independent of any annotated key/mode: it answers
    "which 7-note collection do these notes come from", not "which note in
    it feels like home" -- the two problems this whole analysis has been
    arguing should be solved separately.
    """
    real_events = [ev for ev in chord_events if ev.root >= 0]
    if not real_events:
        return 0, 0.0
    best_T, best_frac = 0, -1.0
    for T in range(12):
        matched = sum(
            1 for ev in real_events
            if DIATONIC_MAJOR_FAMILY.get((ev.root - T) % 12) == precise_triad_quality(ev.quality)
        )
        frac = matched / len(real_events)
        if frac > best_frac:
            best_T, best_frac = T, frac
    return best_T, best_frac


def illustrate_parent_scale_identification():
    """
    Validates identify_best_parent_scale() against the GT-implied collection
    (song's own tonic if major-annotated, tonic+3 if minor-annotated) across
    all 909 songs -- does the simpler 12-way problem actually solve
    correctly on its own, using nothing but chord content (no key_audio.txt
    lookup at all)?
    """
    from harmonia.data.pop909_parser import POP909Parser, parse_harte_label

    parser = POP909Parser(POP909_DIR)
    songs = parser.parse_all(require_audio=False)

    agree, total = 0, 0
    match_fracs = []
    per_mode_agree = {"major": [0, 0], "minor": [0, 0]}

    for song in songs:
        key_path = POP909_DIR / song.song_id / "key_audio.txt"
        if not key_path.exists() or not song.chord_events:
            continue
        line = open(key_path).readline().split()
        if len(line) < 3:
            continue
        parsed = parse_harte_label(line[2])
        if parsed is None:
            continue
        tonic, key_quality = parsed
        mode = _TQ_TO_KEY_MODE.get(quality_bucket(key_quality), "major")
        gt_collection_tonic = tonic if mode == "major" else (tonic + 3) % 12

        best_T, frac = identify_best_parent_scale(song.chord_events)
        match_fracs.append(frac)
        total += 1
        per_mode_agree[mode][1] += 1
        if best_T == gt_collection_tonic:
            agree += 1
            per_mode_agree[mode][0] += 1

    print(f"Parent-scale identification agrees with GT-implied collection: "
          f"{agree}/{total} songs ({agree/total:.1%})")
    for mode in ["major", "minor"]:
        a, t = per_mode_agree[mode]
        print(f"  {mode}-annotated: {a}/{t} ({a/max(t,1):.1%})")
    print(f"  mean within-song diatonic-triad-match fraction at the best T: "
          f"{np.mean(match_fracs):.1%}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(match_fracs, bins=30, color="#1f77b4")
    ax.set_xlabel("fraction of a song's chords matching the best-fit parent scale's diatonic triads")
    ax.set_ylabel("number of songs")
    ax.set_title(f"Mode-agnostic parent-scale identification, all 909 songs\n"
                 f"agrees with GT-implied collection in {agree}/{total} songs ({agree/total:.1%})")
    fig.tight_layout()
    out = PLOT_ROOT / "parent_scale_identification.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def illustrate_atomic_bigrams(top_k: int = 15):
    """
    The fully mode-agnostic pooling: label every chord purely by its scale
    POSITION within its song's own best-fit parent-scale collection
    (identify_best_parent_scale(), NOT the GT tonic/mode) -- Ionian-style
    roman numerals are used as position labels by convention only, they do
    NOT imply an Ionian/major modal centre. Pools major- and minor-annotated
    songs (and, implicitly, any other mode -- Dorian, Mixolydian, etc. --
    since none of this depends on which position is felt as home) into ONE
    table. Chords whose root isn't even a member of the identified
    collection are tracked separately as "cross-scale" -- candidate real
    modulations, not just modal reinterpretation.
    """
    from harmonia.data.pop909_parser import POP909Parser

    parser = POP909Parser(POP909_DIR)
    songs = parser.parse_all(require_audio=False)

    bigrams = Counter()
    cross_scale_count = 0
    total_transitions = 0

    for song in songs:
        if not song.chord_events:
            continue
        best_T, _ = identify_best_parent_scale(song.chord_events)
        collection_pcs = {(best_T + iv) % 12 for iv in MAJOR_FAMILY}

        real_events = [ev for ev in song.chord_events if ev.root >= 0]
        for a, b in zip(real_events, real_events[1:]):
            if a.root == b.root and a.quality == b.quality:
                continue
            total_transitions += 1
            if a.root not in collection_pcs or b.root not in collection_pcs:
                cross_scale_count += 1
                continue
            deg_a, deg_b = (a.root - best_T) % 12, (b.root - best_T) % 12
            key_a = (_INTERVAL_TO_ROMAN[deg_a], quality_bucket(a.quality))
            key_b = (_INTERVAL_TO_ROMAN[deg_b], quality_bucket(b.quality))
            bigrams[(key_a, key_b)] += 1

    total = sum(bigrams.values())
    print(f"Atomic (mode-agnostic) bigrams: {total} within-collection transitions, "
          f"{cross_scale_count} cross-scale transitions "
          f"({cross_scale_count/total_transitions:.1%} of {total_transitions} total)")
    top_items = bigrams.most_common(top_k)
    for (ka, kb), c in top_items:
        print(f"  {ka[0]}{ka[1]:<5s} -> {kb[0]}{kb[1]:<5s}   {c:6d}  ({c/total:.2%})")

    fig, ax = plt.subplots(figsize=(9, 7))
    labels = [f"{ka[0]}{ka[1]} -> {kb[0]}{kb[1]}" for (ka, kb), _ in reversed(top_items)]
    values = [c / total for _, c in reversed(top_items)]
    ax.barh(labels, values, color="#1f77b4")
    for i, v in enumerate(values):
        ax.text(v + 0.0005, i, f"{v:.2%}", va="center", fontsize=8)
    ax.set_xlabel("share of within-collection transitions (909 songs, fully pooled, mode-agnostic)")
    ax.set_title(f"Atomic bigrams: pooled by best-fit parent scale, NOT by annotated major/minor\n"
                 f"(n={total} within-collection; {cross_scale_count/total_transitions:.1%} of all "
                 f"transitions are cross-scale, tracked separately)")
    fig.tight_layout()
    out = PLOT_ROOT / "atomic_bigrams.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


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
