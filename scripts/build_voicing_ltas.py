"""Build LTAS chroma distributions from synthesised chord voicings.

For every (quality × root) combination we generate multiple voicings:
  - open / close / drop-2 / drop-3 / shell voicings
  - bass doubling on/off
  - multiple octave registers
  - multiple GM timbres (piano, electric piano, organ, guitar, strings)
  - two soundfonts

All rendered at root = C (MIDI 60), then the LTAS chroma is extracted and
root-shifted by -root_pc → index 0 always = root.

The resulting {quality: [n_vecs × 12]} dictionary is pooled with the corpus
data in chord_tree_ltas.py to build richer distributions for rare qualities.

Saves to: data/cache/chord_tree_voicing_ltas.npz
  Keys: exact_{quality}_vecs   (float32, shape (N,12))

Then rebuilds all node distributions and saves to data/cache/chord_tree_ltas.npz,
overwriting the corpus-only version.

Usage:
    .venv/bin/python scripts/build_voicing_ltas.py
    .venv/bin/python scripts/build_voicing_ltas.py --n-roots 12  # all 12 roots
    .venv/bin/python scripts/build_voicing_ltas.py --no-rebuild   # just generate, don't overwrite tree cache
"""
from __future__ import annotations
import argparse, sys, tempfile, warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import librosa
import numpy as np
import pretty_midi

from build_accomp_audio_hard import render_to_array, SOUNDFONTS
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig

VOICING_CACHE = REPO / "data" / "cache" / "chord_tree_voicing_ltas.npz"
TREE_CACHE    = REPO / "data" / "cache" / "chord_tree_ltas.npz"

# ── chord tree ──────────────────────────────────────────────────────────────
TREE = {
    "major":      {"majT":["maj","6"], "maj7":["maj7"], "dom7":["dom7","dom7alt"]},
    "minor":      {"minT":["min","m6"], "min7":["min7"], "minmaj7":["minmaj7"]},
    "diminished": {"dimT":["dim"], "dim7":["dim7"], "m7b5":["m7b5"]},
    "augmented":  {"augT":["aug"], "aug7":["aug7"], "augmaj7":["augmaj7"]},
    "suspended":  {"susT":["sus2","sus4"], "7sus4":["7sus4"]},
}
BASE7_FAM = {b7:fam for fam,ch in TREE.items() for b7 in ch}
EXACT_FAM = {ex:fam for fam,ch in TREE.items() for b7,exs in ch.items() for ex in exs}
EXACT_B7  = {ex:b7  for fam,ch in TREE.items() for b7,exs in ch.items() for ex in exs}

# ── chord tone definitions (semitones from root) ─────────────────────────────
# Each quality: list of intervals from root.
# dom7alt = b7 + b9 or #9 variant (we generate both).
CHORD_INTERVALS: dict[str, list[int]] = {
    "maj":     [0, 4, 7],
    "6":       [0, 4, 7, 9],
    "maj7":    [0, 4, 7, 11],
    "dom7":    [0, 4, 7, 10],
    # dom7alt: same chord tones as dom7; "alt" means variable tensions (b9/#9/b13)
    # that shift per voicing and add noise to LTAS. Treat as 7th quality only.
    "dom7alt": [0, 4, 7, 10],
    "min":     [0, 3, 7],
    "m6":      [0, 3, 7, 9],
    "min7":    [0, 3, 7, 10],
    "minmaj7": [0, 3, 7, 11],
    "dim":     [0, 3, 6],
    "dim7":    [0, 3, 6, 9],
    "m7b5":    [0, 3, 6, 10],
    "aug":     [0, 4, 8],
    "aug7":    [0, 4, 8, 10],
    "augmaj7": [0, 4, 8, 11],
    "sus2":    [0, 2, 7],
    "sus4":    [0, 5, 7],
    "7sus4":   [0, 5, 7, 10],
}

# GM programs to cycle through (piano, ep, organ, guitar, strings)
GM_PROGRAMS = [0, 4, 5, 16, 24, 48, 52]


# ── voicing generator ────────────────────────────────────────────────────────

def _close_voicing(intervals: list[int], root_midi: int) -> list[int]:
    """Pack all tones into one octave above root."""
    return sorted(root_midi + iv % 12 for iv in intervals)


def _open_voicing(intervals: list[int], root_midi: int) -> list[int]:
    """Spread tones across ~2 octaves, root at bottom."""
    notes = sorted(set(iv % 12 for iv in intervals))
    result = [root_midi + notes[0]]
    for i, iv in enumerate(notes[1:], 1):
        prev = result[-1] % 12
        step = (iv - prev) % 12
        # if step is small (≤5), push up an octave for openness
        octave_bump = 12 if step <= 5 and i > 1 else 0
        result.append(result[-1] + step + octave_bump)
    return result


def _drop2(notes: list[int]) -> list[int]:
    """Drop the 2nd-highest note down an octave."""
    if len(notes) < 3:
        return notes
    s = sorted(notes)
    s[-2] -= 12
    return sorted(s)


def _drop3(notes: list[int]) -> list[int]:
    """Drop the 3rd-highest note down an octave."""
    if len(notes) < 4:
        return notes
    s = sorted(notes)
    s[-3] -= 12
    return sorted(s)


def _shell(intervals: list[int], root_midi: int) -> list[int]:
    """Root + 3rd/4th + 7th only (jazz shell voicing)."""
    keep = {0}  # root always
    # find the 3rd or 4th (3, 4, 5 semitones)
    for iv in intervals:
        if iv % 12 in (3, 4, 5): keep.add(iv % 12); break
    # find the 7th (10 or 11)
    for iv in intervals:
        if iv % 12 in (9, 10, 11): keep.add(iv % 12); break
    notes = sorted(root_midi + k for k in keep)
    return notes if len(notes) >= 2 else _close_voicing(intervals, root_midi)


def _with_bass_doubling(notes: list[int]) -> list[int]:
    """Add root an octave below the lowest note."""
    return [min(notes) - 12] + notes


def generate_voicings(quality: str, root_pc: int,
                      octave_centers: tuple[int, ...] = (48, 55, 60, 65)
                      ) -> list[list[int]]:
    """
    Return all voicing variants for (quality, root_pc).
    Root is expressed as MIDI note = octave_center + root_pc.
    Each variant is a list of MIDI pitches.
    """
    ivs = CHORD_INTERVALS[quality]
    voicings = []
    for center in octave_centers:
        root_midi = center + root_pc
        close  = _close_voicing(ivs, root_midi)
        open_  = _open_voicing(ivs, root_midi)
        shell  = _shell(ivs, root_midi)
        variants = [close, open_, shell]
        if len(close) >= 3:
            variants.append(_drop2(close))
        if len(close) >= 4:
            variants.append(_drop3(close))
        for v in list(variants):
            variants.append(_with_bass_doubling(v))
        voicings.extend(variants)

    # deduplicate
    seen: set = set()
    unique = []
    for v in voicings:
        k = tuple(v)
        if k not in seen:
            seen.add(k); unique.append(v)
    return unique


# ── audio rendering ──────────────────────────────────────────────────────────

def _notes_to_pm(notes: list[int], program: int, duration: float = 3.0) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=60)
    inst = pretty_midi.Instrument(program=program)
    for pitch in notes:
        if 0 <= pitch <= 127:
            inst.notes.append(pretty_midi.Note(
                velocity=90, pitch=pitch, start=0.0, end=duration))
    pm.instruments.append(inst)
    return pm


def _ltas_cqt(audio: np.ndarray, sr: int, hop: int = 512) -> np.ndarray:
    raw = librosa.feature.chroma_cqt(y=audio, sr=sr, bins_per_octave=36, hop_length=hop)
    ltas = raw.mean(axis=1, keepdims=True)
    ltas = np.where(ltas < 1e-9, 1.0, ltas)
    chroma = (raw / ltas).mean(axis=1)   # (12,) LTAS mean
    n = np.linalg.norm(chroma)
    return chroma / n if n > 1e-9 else chroma


def render_and_extract(notes: list[int], root_pc: int, program: int,
                       sf_name: str, renderer: MIDIRenderer,
                       duration: float = 3.0) -> np.ndarray | None:
    """
    Render a chord voicing, extract LTAS chroma, root-shift to root=0.
    Returns (12,) float32 or None on failure.
    """
    try:
        pm = _notes_to_pm(notes, program, duration)
        audio, sr = render_to_array(renderer, pm, sf_name, reverb=False)
        # trim leading/trailing silence
        audio = audio.astype(float)
        audio = audio[int(0.05*sr):int((duration-0.1)*sr)]
        chroma = _ltas_cqt(audio, sr)                     # already L2-normed
        shifted = np.roll(chroma, -root_pc)                # root → index 0
        n = np.linalg.norm(shifted)
        return (shifted / n).astype(np.float32) if n > 1e-9 else None
    except Exception:
        return None


# ── main collection loop ─────────────────────────────────────────────────────

def build_voicing_vectors(n_roots: int, seed: int) -> dict[str, list[np.ndarray]]:
    """
    Generate voicing vectors for all 18 exact qualities × n_roots roots.
    Returns {quality: [L2-normed root-shifted (12,) vectors]}.
    """
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    rng = np.random.default_rng(seed)

    roots = list(range(n_roots))          # 0=C, 1=Db, ..., 11=B
    qualities = list(CHORD_INTERVALS.keys())
    by_exact: dict[str, list[np.ndarray]] = defaultdict(list)

    total = len(qualities) * n_roots
    done  = 0
    for qi, quality in enumerate(qualities):
        for root_pc in roots:
            done += 1
            print(f"\r  [{done:3d}/{total}] {quality:10s} root={root_pc:2d}", end="", flush=True)
            voicings = generate_voicings(quality, root_pc)
            for v in voicings:
                for sf in SOUNDFONTS:
                    for program in rng.choice(GM_PROGRAMS, size=3, replace=False):
                        vec = render_and_extract(v, root_pc, int(program), sf, renderer)
                        if vec is not None:
                            by_exact[quality].append(vec)
    print()
    return by_exact


# ── distribution aggregation ─────────────────────────────────────────────────

def aggregate_distributions(by_exact: dict[str, list[np.ndarray]],
                             corpus_dist: dict | None = None) -> dict:
    """
    Merge voicing vectors with optional corpus vectors, then compute
    μ/σ for all 37 tree nodes (family, base7, exact).
    Mirrors the logic in chord_tree_ltas.build_distributions().
    """
    out: dict[str, np.ndarray] = {}
    for fam, b7_dict in TREE.items():
        fam_vecs = []
        for b7, exacts in b7_dict.items():
            b7_vecs = []
            for ex in exacts:
                vecs = list(by_exact.get(ex, []))
                # merge in corpus vectors if available
                if corpus_dist is not None:
                    n_corpus = int(corpus_dist.get(f"exact_{ex}_n", np.array([0]))[0])
                    # reconstruct individual corpus vectors is not possible —
                    # but we can augment the distribution by treating the corpus
                    # mean as extra synthetic observations (weight by n_corpus)
                    # Better: just save voicing vecs separately and let the caller merge.
                if len(vecs) < 2:
                    vecs = [np.zeros(12, np.float32), np.ones(12, np.float32)/np.sqrt(12)]
                arr = np.stack(vecs).astype(np.float32)
                out[f"exact_{ex}_mu"]  = arr.mean(0)
                out[f"exact_{ex}_std"] = arr.std(0) + 1e-4
                out[f"exact_{ex}_n"]   = np.array([len(vecs)])
                b7_vecs.extend(vecs)
            arr = np.stack(b7_vecs).astype(np.float32)
            out[f"b7_{b7}_mu"]  = arr.mean(0)
            out[f"b7_{b7}_std"] = arr.std(0) + 1e-4
            out[f"b7_{b7}_n"]   = np.array([len(b7_vecs)])
            fam_vecs.extend(b7_vecs)
        arr = np.stack(fam_vecs).astype(np.float32)
        out[f"fam_{fam}_mu"]  = arr.mean(0)
        out[f"fam_{fam}_std"] = arr.std(0) + 1e-4
        out[f"fam_{fam}_n"]   = np.array([len(fam_vecs)])
    return out


def merge_with_corpus(voicing_by_exact: dict[str, list[np.ndarray]],
                      corpus_cache: Path) -> dict[str, list[np.ndarray]]:
    """
    If corpus cache exists, load per-quality corpus vectors and pool them
    with the voicing vectors. The corpus cache stores only μ/σ/n (not
    individual vectors), so we synthesise n pseudo-samples by sampling
    from the diagonal Gaussian — this is an approximation but keeps the
    variance calibrated.
    """
    if not corpus_cache.exists():
        return voicing_by_exact
    d = np.load(corpus_cache)
    combined = dict(voicing_by_exact)
    rng = np.random.default_rng(0)
    for ex in CHORD_INTERVALS:
        key_mu  = f"exact_{ex}_mu"
        key_std = f"exact_{ex}_std"
        key_n   = f"exact_{ex}_n"
        if key_mu not in d.files: continue
        mu  = d[key_mu]; std = d[key_std]
        n   = int(d[key_n][0])
        if n < 2: continue
        # draw n samples from the fitted diagonal Gaussian
        samples = rng.normal(mu, std, size=(n, 12)).astype(np.float32)
        # L2-normalise each sample
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        samples = samples / np.where(norms < 1e-9, 1.0, norms)
        combined.setdefault(ex, [])
        combined[ex].extend(list(samples))
    return combined


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-roots",    type=int, default=1,
                    help="roots to generate (default 1=C only; root-shifting makes others redundant)")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--no-rebuild", action="store_true",
                    help="save voicing cache but do NOT overwrite chord_tree_ltas.npz")
    ap.add_argument("--no-corpus-merge", action="store_true",
                    help="use voicing vectors only (ignore corpus distributions)")
    args = ap.parse_args()

    print(f"Generating voicing vectors — {args.n_roots} root(s) × 18 qualities...")
    print(f"  (will render ~{args.n_roots * 18 * 10 * len(SOUNDFONTS) * 3} audio clips)")
    by_exact = build_voicing_vectors(args.n_roots, args.seed)

    # report counts
    print("\nVectors per quality (before corpus merge):")
    for q in CHORD_INTERVALS:
        print(f"  {q:12s}: {len(by_exact.get(q, []))}")

    # save raw voicing vectors
    save_dict = {f"exact_{q}_vecs": np.stack(v).astype(np.float32)
                 for q, v in by_exact.items() if v}
    VOICING_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(VOICING_CACHE, **save_dict)
    print(f"\n→ Voicing vectors saved to {VOICING_CACHE}")

    if not args.no_rebuild:
        # merge with corpus
        if not args.no_corpus_merge:
            print("Merging with corpus distributions...")
            by_exact_merged = merge_with_corpus(by_exact, TREE_CACHE)
        else:
            by_exact_merged = by_exact

        print("Aggregating distributions for all 37 tree nodes...")
        dist = aggregate_distributions(by_exact_merged)

        print("\nFinal node sample counts:")
        for fam in TREE:
            n = int(dist[f"fam_{fam}_n"][0])
            print(f"  {fam:12s}: n={n}")
            for b7 in TREE[fam]:
                nb = int(dist[f"b7_{b7}_n"][0])
                print(f"    {b7:12s}: n={nb}")
                for ex in TREE[fam][b7]:
                    ne = int(dist[f"exact_{ex}_n"][0])
                    print(f"      {ex:12s}: n={ne}")

        np.savez(TREE_CACHE, **dist)
        print(f"\n→ Updated chord tree distributions saved to {TREE_CACHE}")
        print("  Run chord_tree_ltas.py --rebuild=False to re-plot with new distributions.")
        print("  Run diagnose_exact_tree.py to evaluate the updated tree search.")
