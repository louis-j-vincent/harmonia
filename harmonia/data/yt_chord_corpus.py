"""yt_chord_corpus.py — build a chord-training corpus from YouTube + iReal Pro GT.

Each entry is a (YouTube video ID, iReal Pro URL, song title) triple.
For each entry we:
  1. Download audio via yt-dlp (to a local cache directory).
  2. Run infer_chords_v1 on the audio to get an inferred chord chart.
  3. Align the iReal Pro GT to the inferred chart via irealb_aligner.
  4. Extract per-bar beat-level features (same pathway as chord_pipeline_v1).
  5. Label each feature vector with the iReal GT root + quality.

Output: list of record dicts, or a numpy archive (.npz) with arrays.

Usage example:
    from harmonia.data.yt_chord_corpus import build_corpus, load_corpus

    entries = [
        {"video_id": "YVedK1VUfLM", "title": "Autumn Leaves",
         "irealb_url": "irealb://..."},
    ]
    records = build_corpus(entries, cache_dir=Path("data/cache/yt_corpus"))
    np.savez("data/cache/yt_corpus/corpus.npz", **pack_arrays(records))
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from harmonia.models.chord_pipeline_v1 import (
    BeatFeatures,
    _chroma88,
    _norm_blocks,
    _reg_raw,
    extract_beat_features,
)

logger = logging.getLogger(__name__)

# 7-class quality scheme (maj/min/dom/hdim/dim/aug/sus) — finer than the
# existing 5-class model (which merges dom→major, hdim→diminished).
QUALITIES = ["maj", "min", "dom", "hdim", "dim", "aug", "sus"]
QUALITY_IDX = {q: i for i, q in enumerate(QUALITIES)}


# ── iReal quality tail → 7-class family ──────────────────────────────────────
# Mirrors tab_aligner._family() but also maps N.C. / unknown.

_MAJ_TAILS  = {"", "^7", "maj7", "M", "M7", "6", "maj9", "^9", "add9",
               "maj11", "^11", "maj13", "^13", "2", "5"}
_MIN_TAILS  = {"-", "m", "min", "-7", "m7", "min7", "-9", "m9", "-11",
               "m11", "-13", "m13", "-6", "m6", "-^7", "mMaj7", "minMaj7"}
_DOM_TAILS  = {"7", "9", "11", "13", "+7", "aug7", "7b9", "7#9", "7alt",
               "7b5", "7#5", "7#11", "7b13", "9#11", "9b5", "13b9", "13#11"}
_HDIM_TAILS = {"h7", "m7b5", "ø7", "ø"}
_DIM_TAILS  = {"o", "dim", "o7", "dim7", "°", "°7"}
_AUG_TAILS  = {"+", "aug", "+^7", "augMaj7"}
_SUS_TAILS  = {"sus", "sus4", "sus2", "7sus", "7sus4"}


def quality_family(tail: str) -> str | None:
    """iReal quality tail → 7-class family string, or None for N.C."""
    tail = tail.strip()
    if tail in _MAJ_TAILS:  return "maj"
    if tail in _MIN_TAILS:  return "min"
    if tail in _DOM_TAILS:  return "dom"
    if tail in _HDIM_TAILS: return "hdim"
    if tail in _DIM_TAILS:  return "dim"
    if tail in _AUG_TAILS:  return "aug"
    if tail in _SUS_TAILS:  return "sus"
    # fallback heuristics (same as tab_aligner._family)
    if tail.startswith("-") or tail.startswith("m"): return "min"
    if tail.startswith("o") or tail.startswith("°"): return "dim"
    if tail.startswith("+"):                          return "aug"
    if tail.startswith("h"):                          return "hdim"
    return None   # N.C., unknown


# ── segment features (matches _FamilyClassifier.predict pathway exactly) ─────

def seg_feature(
    onset_b: np.ndarray, note_b: np.ndarray,
    b0: int, b1: int, root: int,
) -> np.ndarray:
    """48-dim root-shifted, L2-normed chroma feature for beats b0:b1.

    Matches the feature computed inside _FamilyClassifier.predict():
      _norm_blocks(hstack([roll(ch_on,-root), roll(ch_nt,-root),
                           roll(bass,-root), roll(treble,-root)]))
    """
    seg_on = onset_b[b0:b1].sum(0)
    seg_nt = note_b[b0:b1].sum(0)
    ch_on = _reg_raw(seg_on)
    ch_nt = _reg_raw(seg_nt)
    seg_bs = _reg_raw(seg_on, 0, 52)
    seg_tr = _reg_raw(seg_on, 60, 200)
    rr = lambda c: np.roll(c, -root)
    return _norm_blocks(np.hstack([rr(ch_on), rr(ch_nt), rr(seg_bs), rr(seg_tr)])).astype(np.float32)


def seg_feature_abs(
    onset_b: np.ndarray, note_b: np.ndarray,
    b0: int, b1: int,
) -> np.ndarray:
    """48-dim absolute chroma (NOT root-shifted) for root classification."""
    seg_on = onset_b[b0:b1].sum(0)
    seg_nt = note_b[b0:b1].sum(0)
    ch_on = _reg_raw(seg_on)
    ch_nt = _reg_raw(seg_nt)
    seg_bs = _reg_raw(seg_on, 0, 52)
    seg_tr = _reg_raw(seg_on, 60, 200)
    return _norm_blocks(np.hstack([ch_on, ch_nt, seg_bs, seg_tr])).astype(np.float32)


# ── boundary-bleed-free clipped pooling (2026-07-16) ─────────────────────────
# The (t0,t1)->beat-range snap used by every corpus builder
# (`b0=searchsorted(bt,t0,"right")-1; b1=searchsorted(bt,t1,"right")`) pools
# WHOLE beats, so the beat that merely *contains* t1 is summed in full — pulling
# ~1 beat (mean ~310ms, up to ~476ms) of the NEXT chord into this chord's
# feature. See docs/known_issues.md "boundary bleed". These helpers pool
# frame-level activations clipped EXACTLY to [t0,t1) — no beat grid, zero bleed —
# and are drop-in replacements for seg_feature[_abs] given frame-level acts
# (frame_times, onset_probs, note_probs) instead of beat-pooled onset_b/note_b.

def _clip_pool(frame_times: np.ndarray, onset_f: np.ndarray, note_f: np.ndarray,
               t0: float, t1: float):
    """Sum frame activations whose frame centre lies in [t0,t1). Returns
    (seg_on(88,), seg_nt(88,)) or (None, None) if no frame falls in the span."""
    m = (frame_times >= t0) & (frame_times < t1)
    if not m.any():
        return None, None
    return onset_f[m].sum(0), note_f[m].sum(0)


def seg_feature_clipped(frame_times, onset_f, note_f, t0, t1, root):
    """Root-shifted 48-d feature pooled over frames exactly in [t0,t1)."""
    seg_on, seg_nt = _clip_pool(frame_times, onset_f, note_f, t0, t1)
    if seg_on is None:
        return None
    ch_on = _reg_raw(seg_on); ch_nt = _reg_raw(seg_nt)
    seg_bs = _reg_raw(seg_on, 0, 52); seg_tr = _reg_raw(seg_on, 60, 200)
    rr = lambda c: np.roll(c, -root)
    return _norm_blocks(np.hstack([rr(ch_on), rr(ch_nt), rr(seg_bs), rr(seg_tr)])).astype(np.float32)


def seg_feature_abs_clipped(frame_times, onset_f, note_f, t0, t1):
    """Absolute (non-root-shifted) 48-d feature pooled over frames in [t0,t1)."""
    seg_on, seg_nt = _clip_pool(frame_times, onset_f, note_f, t0, t1)
    if seg_on is None:
        return None
    ch_on = _reg_raw(seg_on); ch_nt = _reg_raw(seg_nt)
    seg_bs = _reg_raw(seg_on, 0, 52); seg_tr = _reg_raw(seg_on, 60, 200)
    return _norm_blocks(np.hstack([ch_on, ch_nt, seg_bs, seg_tr])).astype(np.float32)


# ── audio download ─────────────────────────────────────────────────────────────

def download_audio(video_id: str, out_dir: Path, *, force: bool = False) -> Path:
    """Download a YouTube video as a mono 22050 Hz wav via yt-dlp + ffmpeg.

    Downloads best audio stream via yt-dlp, then converts to wav via ffmpeg
    explicitly (yt-dlp's postprocessor-args are unreliable without a JS runtime).
    Returns path to the .wav file.  Skips if already cached (unless force=True).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{video_id}.wav"
    if wav_path.exists() and not force:
        return wav_path

    url = f"https://www.youtube.com/watch?v={video_id}"
    raw_path = out_dir / f"{video_id}.%(ext)s"

    # Locate yt-dlp: prefer venv's bin over system PATH
    import shutil
    ytdlp = shutil.which("yt-dlp") or str(Path(sys.executable).parent / "yt-dlp")

    # Step 1: download best audio stream (webm/opus/m4a)
    dl_cmd = [
        ytdlp, "-x", "--audio-quality", "0",
        "-o", str(raw_path), url,
    ]
    logger.info("Downloading %s ...", url)
    r = subprocess.run(dl_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"yt-dlp failed for {video_id}:\n{r.stderr}")

    # Step 2: find the downloaded file (any extension)
    candidates = list(out_dir.glob(f"{video_id}.*"))
    src = next((p for p in candidates if p.suffix != ".wav"), None)
    if src is None:
        raise FileNotFoundError(f"yt-dlp produced no file for {video_id}")

    # Step 3: convert to mono 22050 Hz wav via ffmpeg
    ff_cmd = ["ffmpeg", "-y", "-i", str(src), "-ar", "22050", "-ac", "1", str(wav_path)]
    r2 = subprocess.run(ff_cmd, capture_output=True, text=True)
    if r2.returncode != 0 or not wav_path.exists():
        raise RuntimeError(f"ffmpeg conversion failed for {video_id}:\n{r2.stderr[-500:]}")

    # Remove intermediate file
    src.unlink(missing_ok=True)
    return wav_path


# ── per-entry feature extraction ──────────────────────────────────────────────

_NOTE_PC = {n: i for i, n in enumerate(
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
)}


def _chart_chord_to_p(c: dict) -> dict:
    """Convert a ChordChart chord (Harte label) to the p_chords format
    expected by irealb_aligner.dedup_inferred()."""
    label = c.get("label", "")
    if ":" in label:
        root_str, quality = label.split(":", 1)
    else:
        root_str = label; quality = "maj"
    pc = _NOTE_PC.get(root_str, -1)
    return {
        "root": pc,
        "lv": {"seventh": {"q": ":" + quality}},
        "t0": c["start_s"],
        "t1": c["end_s"],
        "label": label,
    }


def extract_records(
    audio_path: Path,
    irealb_url: str,
    *,
    cache_dir: Path | None = None,
    min_beats: int = 2,
) -> list[dict]:
    """Extract (feature, root, quality) records from one (audio, iReal) pair.

    Returns list of dicts:
      {feat48: np.ndarray(48,), feat48_abs: np.ndarray(48,),
       root: int, quality: str, quality_idx: int,
       t0: float, t1: float, label: str, match: str}

    Alignment mismatches are included (with match='mismatch') so the corpus
    builder can filter them if needed.  Empty bars and N.C. slots are skipped.
    """
    from harmonia.models.chord_pipeline_v1 import infer_chords_v1
    from harmonia.irealb_aligner import align_irealb_to_inferred

    # Step 1: extract beat-level features (reuse existing pipeline steps 1-4)
    bf = extract_beat_features(audio_path, cache_dir=cache_dir)

    # Step 2: infer chart (needed for alignment)
    chart = infer_chords_v1(audio_path, cache_dir=cache_dir)
    p_chords = [_chart_chord_to_p(c) for c in chart.chords]

    # Step 3: parse iReal URL → MMAChart, then align
    import urllib.parse
    from pyRealParser import Tune
    from harmonia.data.ireal_corpus import tune_to_mma
    decoded = urllib.parse.unquote(irealb_url)
    tunes = Tune.parse_ireal_url(decoded)
    if not tunes:
        logger.warning("Could not parse iReal URL for %s", audio_path.name)
        return []
    mma_chart = tune_to_mma(tunes[0])

    # Step 4: audio features needed by aligner (Bloc C) and by feature extraction.
    import librosa, soundfile as sf
    y, sr = sf.read(audio_path)
    y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
    hop = 512
    cqt_raw   = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36, hop_length=hop)
    cqt_times = librosa.frames_to_time(np.arange(cqt_raw.shape[1]), sr=sr, hop_length=hop)
    # High-freq onset flux for Bloc C boundary refinement
    y_hf       = librosa.effects.preemphasis(y, coef=0.97)
    onset_env  = librosa.onset.onset_strength(y=y_hf, sr=sr, hop_length=hop)
    onset_flux = np.convolve(onset_env, np.ones(3) / 3, mode="same").astype(np.float32)
    onset_times = librosa.frames_to_time(np.arange(len(onset_flux)), sr=sr, hop_length=hop)

    result = align_irealb_to_inferred(
        mma_chart, p_chords,
        onset_flux_times=onset_times,
        onset_flux=onset_flux,
    )
    if not result.chords:
        logger.warning("Empty alignment for %s", audio_path.name)
        return []

    # Step 5: for each aligned iReal chord, extract features at the right beats
    from harmonia.tab_aligner import _parse_ireal as _pi  # noqa: F401
    onset_b   = bf.onset_b
    note_b    = bf.note_b
    beat_times = bf.beat_times

    records: list[dict] = []
    for ch in result.chords:
        t0, t1 = ch["t0"], ch["t1"]
        label  = ch["label"]
        match  = ch.get("match", "")

        # parse root + quality tail from iReal label
        pc, qtail = _pi(label)
        if pc < 0:
            continue  # N.C. or unparseable
        fam = quality_family(qtail)
        if fam is None:
            continue

        # find beats in [t0, t1)
        b0 = int(np.searchsorted(beat_times, t0, side="right")) - 1
        b1 = int(np.searchsorted(beat_times, t1, side="right"))
        b0 = max(b0, 0); b1 = min(b1, len(onset_b))
        if b1 - b0 < min_beats:
            continue

        feat48     = seg_feature(onset_b, note_b, b0, b1, pc)
        feat48_abs = seg_feature_abs(onset_b, note_b, b0, b1)

        # CQT chroma for this segment: mean over frames in [t0, t1), root-shifted
        cqt_mask = (cqt_times >= t0) & (cqt_times < t1)
        if cqt_mask.any():
            cqt_seg = cqt_raw[:, cqt_mask].mean(axis=1)
            nn = np.linalg.norm(cqt_seg)
            cqt_seg = (cqt_seg / nn if nn > 1e-9 else cqt_seg).astype(np.float32)
        else:
            cqt_seg = np.zeros(12, np.float32)
        cqt_shifted = np.roll(cqt_seg, -pc).astype(np.float32)   # root-shifted

        records.append({
            "feat48":     feat48,
            "feat48_abs": feat48_abs,
            "feat12_cqt": cqt_shifted,      # root-shifted CQT chroma (12d)
            "feat12_cqt_abs": cqt_seg,      # absolute CQT chroma (12d)
            "root":       int(pc % 12),
            "quality":    fam,
            "quality_idx": QUALITY_IDX[fam],
            "t0":         float(t0),
            "t1":         float(t1),
            "label":      label,
            "match":      match,
        })

    logger.info(
        "  %s: %d records  (exact=%.0f%% family=%.0f%% mismatch=%.0f%%)",
        audio_path.name, len(records),
        100 * result.exact_frac, 100 * result.family_frac, 100 * result.mismatch_frac,
    )
    return records


# ── corpus builder ─────────────────────────────────────────────────────────────

def build_corpus(
    entries: list[dict],
    cache_dir: Path,
    *,
    skip_mismatches: bool = False,
    force_download: bool = False,
    delete_wav_after: bool = True,
) -> list[dict]:
    """Build a training corpus from a list of (video_id, irealb_url, title) entries.

    Args:
        entries:         List of dicts with keys: video_id, irealb_url, title.
        cache_dir:       Directory for downloaded audio + Basic Pitch cache.
        skip_mismatches: If True, exclude alignment-mismatch records.
        force_download:  Re-download audio even if already cached.
        delete_wav_after: Delete WAV file after feature extraction to save disk.
                          BP activations remain cached in bp_cache/. Default True.

    Returns list of record dicts (see extract_records for schema).
    """
    cache_dir = Path(cache_dir)
    audio_dir = cache_dir / "audio"
    bp_cache  = cache_dir / "bp_cache"
    audio_dir.mkdir(parents=True, exist_ok=True)
    bp_cache.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    for i, entry in enumerate(entries):
        vid   = entry["video_id"]
        url   = entry["irealb_url"]
        title = entry.get("title", vid)
        print(f"[{i+1}/{len(entries)}] {title} ({vid})", flush=True)
        try:
            wav = download_audio(vid, audio_dir, force=force_download)
        except Exception as exc:
            logger.warning("  download failed: %s", exc)
            continue
        try:
            recs = extract_records(wav, url, cache_dir=bp_cache)
        except Exception as exc:
            logger.warning("  feature extraction failed: %s", exc)
            continue
        finally:
            # Delete WAV after extraction to keep disk usage low.
            # BP activations are already in bp_cache and don't need the WAV.
            if delete_wav_after and wav.exists():
                wav.unlink(missing_ok=True)

        for r in recs:
            r["song_id"] = vid
        if skip_mismatches:
            recs = [r for r in recs if r["match"] in ("exact", "family")]
        all_records.extend(recs)
        print(f"  → {len(recs)} records  (corpus total: {len(all_records)})", flush=True)

    return all_records


def pack_arrays(records: list[dict]) -> dict:
    """Convert list of record dicts to numpy arrays for .npz storage."""
    has_cqt = "feat12_cqt" in records[0] if records else False
    out = {
        "feat48":      np.stack([r["feat48"]     for r in records]),
        "feat48_abs":  np.stack([r["feat48_abs"] for r in records]),
        "root":        np.array([r["root"]       for r in records], dtype=np.int32),
        "quality_idx": np.array([r["quality_idx"] for r in records], dtype=np.int32),
        "quality":     np.array([r["quality"]    for r in records]),
        "labels":      np.array([r["label"]      for r in records]),
        "match":       np.array([r["match"]      for r in records]),
        "t0":          np.array([r["t0"]         for r in records]),
        "t1":          np.array([r["t1"]         for r in records]),
        "song_id":     np.array([r.get("song_id", "") for r in records]),
        "qualities":   np.array(QUALITIES),
    }
    if has_cqt:
        out["feat12_cqt"]     = np.stack([r["feat12_cqt"]     for r in records])
        out["feat12_cqt_abs"] = np.stack([r["feat12_cqt_abs"] for r in records])
    return out


def load_corpus(path: Path) -> dict:
    """Load a packed corpus from a .npz file."""
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}
