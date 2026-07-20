"""musx_bass.py — music-x-lab bass front-end for the production inference path.

Wraps the pretrained *ISMIR2019 Large-Vocabulary Chord Recognition* model
(music-x-lab, "Chord Structure Decomposition") as a reusable, cached bass
estimator.  It is the strongest single sounding-bass estimator measured on RWC
(0.900 all / 0.744 inversions, full 100-song matched set — beats in-house
NNLS-24 argmax 0.824 all).  See docs/known_issues.md "music-x-lab BASS FRONT-END
DEPLOYED" (2026-07-17).

Role in the pipeline (opt-in, CLAUDE.md rule #6): music-x-lab is the PRIMARY
sounding-bass source; NNLS-24 argmax stays a veto (see routed_bass_pc — the
validated "NNLS root-veto" rule F, +2.0pp over music-x-lab-alone).  Root and
quality remain on the NNLS-24 trained heads; only the bass estimate routes here.

The model is a git clone that ships its own weights (no pip package).  This
module shells out to the clone's `chord_recognition.py` via subprocess and
caches the resulting .lab, mirroring nnls_features.extract_bothchroma's
cache-and-degrade-gracefully contract.  Locate the clone via the
HARMONIA_MUSX_DIR env var; if it (or its weights) are absent, callers get a
RuntimeError with an actionable message and fall back to pure NNLS-24 bass.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent.parent
_CACHE_DIR = REPO / "data" / "cache" / "musx_infer"

# The music-x-lab clone (patched for modern numpy/torch/pyyaml — see
# docs/known_issues.md).  Overridable so a machine can point at its own clone;
# the default is this session's scratchpad clone (ephemeral — a fresh machine
# must `git clone` music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition and
# set HARMONIA_MUSX_DIR, or drop it under harmonia/third_party/).
_DEFAULT_MUSX = Path(
    "/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-"
    "harmonia/a6a4757d-a729-450a-a5be-b9970a43c412/scratchpad/nnls_bass_tools/"
    "ISMIR2019-Large-Vocabulary-Chord-Recognition"
)
_LOCAL_MUSX = (REPO / "harmonia" / "third_party"
               / "ISMIR2019-Large-Vocabulary-Chord-Recognition")

# music-x-lab "submission" Harte quality -> pipeline sev_h token (the spelling
# chord_pipeline_v1 / render_youtube_chart._QUALITY_TO_IREAL understands).  The
# submission vocabulary is fixed (data/submission_chord_list.txt): the 17
# qualities below plus maj/min slash inversions (bass handled separately).
_MUSX_Q_TO_SEV = {
    "maj": "maj", "min": "min", "7": "7", "maj7": "maj7", "min7": "min7",
    "dim": "dim", "dim7": "dim7", "hdim7": "hdim7", "aug": "aug",
    "sus2": "sus2", "sus4": "sus4", "sus4(b7)": "7sus4",
    "9": "9", "min9": "min9", "maj9": "maj9", "11": "dom11", "13": "dom13",
}

# Harte scale-degree bass tokens -> semitone offset from root.
_DEG = {"b2": 1, "2": 2, "b3": 3, "3": 4, "4": 5, "b5": 6, "5": 7,
        "b6": 8, "6": 9, "b7": 10, "7": 11, "b9": 1, "9": 2, "#9": 3,
        "11": 5, "#11": 6, "13": 9}
_NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def musx_dir() -> Path | None:
    """Resolve the music-x-lab clone dir, or None if not available.

    Order: HARMONIA_MUSX_DIR env var, then a repo-local third_party/ copy, then
    the (ephemeral) session scratchpad clone.  A dir counts only if it holds the
    entry script AND the pretrained ensemble weights.
    """
    candidates = []
    env = os.environ.get("HARMONIA_MUSX_DIR")
    if env:
        candidates.append(Path(env))
    candidates += [_LOCAL_MUSX, _DEFAULT_MUSX]
    for d in candidates:
        if (d / "chord_recognition.py").exists() and \
           list((d / "cache_data").glob("*.sdict")):
            return d
    return None


def _parse_root(sym: str) -> int | None:
    """Root pc of a Harte note symbol like 'Bb', 'F#', 'C'."""
    if not sym:
        return None
    pc = _NOTE_PC.get(sym[0].upper())
    if pc is None:
        return None
    for ch in sym[1:]:
        if ch == "#":
            pc += 1
        elif ch == "b":
            pc -= 1
        else:
            break
    return pc % 12


def bass_pc_of_label(label: str) -> int | None:
    """Sounding-bass pc (0–11) of a music-x-lab Harte label, or None for N/X.

    'D:7' -> D (root in bass); 'C:maj/3' -> E (scale-degree); 'C:maj/E' -> E
    (absolute note).  Self-contained (does not import corpus_schema, keeping the
    inference path free of that agent's territory).
    """
    if not label:
        return None
    lab = label.strip()
    if lab in ("N", "X", ""):
        return None
    head = lab.split("/", 1)
    root_sym = head[0].split(":", 1)[0]
    root = _parse_root(root_sym)
    if root is None:
        return None
    if len(head) == 1:
        return root
    tail = head[1].strip()
    if tail in _DEG:
        return (root + _DEG[tail]) % 12
    abs_pc = _parse_root(tail)
    return abs_pc if abs_pc is not None else root


# ── inference (subprocess to the clone, cached per audio file) ───────────────

def _cache_key(audio_path: Path, chord_dict: str) -> str:
    """Stable cache key for an audio file — the file STEM, not its mtime.

    The previous key mixed in ``st_mtime``, which silently defeated the cache in
    production: the server writes every fresh YouTube download to a new temp file
    (``<tmpdir>/<video_id>.<ext>``) with a new mtime, so re-analysing the SAME
    video never hit the cache and always paid the full cold music-x-lab cost
    (docs/known_issues.md DEPLOY-2 caveat).

    A content hash does NOT fix this: yt-dlp's ``bestaudio`` is *not* byte-
    deterministic across downloads of the same video (format/container/timestamp
    variation → different bytes → different hash → still a miss; verified live,
    two downloads of one video gave two distinct hashes).  The **file stem is the
    stable identifier** — for the server path it is the YouTube video id, which is
    invariant across re-downloads, and for a local file it is the filename.  That
    is exactly what identifies "the same song" here, so we key on it.

    Caveat: keying on the stem means a *local* file edited IN PLACE under the same
    name would read a stale cache entry.  For the production (YouTube) path this
    cannot happen (video id ⇒ content); for local dev, clear
    ``data/cache/musx_infer/`` after replacing a file in place.
    """
    return f"{audio_path.stem}_{chord_dict}.lab"



N_MUSX_FOLDS = 5  # len(MODEL_NAMES) in chord_recognition.py — the 5-fold ensemble


def musx_labels(
    audio_path: Path, *, chord_dict: str = "submission",
    use_cache: bool = True,
    progress_cb: "Callable[[int, int, list[tuple[float, float, str]]], None] | None" = None,
) -> list[tuple[float, float, str]]:
    """Run music-x-lab -> list of (t0, t1, Harte-label) segments.

    Cached to data/cache/musx_infer/<stem>_<mtime>_<dict>.lab.  Raises
    RuntimeError (actionable) if the clone/weights are missing so the caller can
    fall back to NNLS-24 bass.

    ``progress_cb(fold_i, n_folds, labels)`` (2026-07-20, progressive-analysis
    screen): if given, polls for the fold-numbered sidecars
    ``chord_recognition.py`` now writes after EACH of its 5 ensemble folds
    (``<cache>.fold1`` .. ``<cache>.fold5``, decoded-so-far average, ~1-2s
    apart) and fires once per new fold as it appears, so a caller can show
    music-x-lab's own chords refining fold-by-fold instead of one silent
    ~10-30s wait. Best-effort: skipped entirely on a cache hit (nothing to
    poll), and any polling error just falls through to the blocking wait for
    the final result — never raises on its own.
    """
    audio_path = Path(audio_path).resolve()  # absolute: subprocess runs in musx cwd
    cache = _CACHE_DIR / _cache_key(audio_path, chord_dict)
    if use_cache and cache.exists():
        return _load_lab(cache)

    mdir = musx_dir()
    if mdir is None:
        raise RuntimeError(
            "music-x-lab bass front-end unavailable: clone not found. `git clone` "
            "https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-"
            "Recognition, apply the numpy/torch/yaml compat patches (see "
            "docs/known_issues.md 2026-07-17), and set HARMONIA_MUSX_DIR or place "
            "it at harmonia/third_party/."
        )

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fold_paths = [Path(f"{cache}.fold{i}") for i in range(1, N_MUSX_FOLDS + 1)]
    for p in fold_paths:  # stale sidecars from a previous crashed run
        p.unlink(missing_ok=True)

    args = [sys.executable, "chord_recognition.py", str(audio_path), str(cache), chord_dict]
    if progress_cb is None:
        r = subprocess.run(args, cwd=str(mdir), capture_output=True, text=True, timeout=900)
        rc, stderr = r.returncode, r.stderr
    else:
        import time as _time
        proc = subprocess.Popen(args, cwd=str(mdir), stdout=subprocess.DEVNULL,
                                 stderr=subprocess.PIPE, text=True)
        seen = 0
        try:
            while proc.poll() is None:
                if seen < len(fold_paths) and fold_paths[seen].exists():
                    seen += 1
                    try:
                        progress_cb(seen, N_MUSX_FOLDS, _load_lab(fold_paths[seen - 1]))
                    except Exception:  # noqa: BLE001 — fold preview is best-effort
                        logger.warning("musx_labels: progress_cb(fold %d) failed",
                                       seen, exc_info=True)
                _time.sleep(0.3)
        finally:
            _, stderr = proc.communicate(timeout=900)
            rc = proc.returncode
            for p in fold_paths:
                p.unlink(missing_ok=True)  # sidecars are transient — never left on disk

    if rc != 0 or not cache.exists():
        raise RuntimeError(f"music-x-lab inference failed (rc={rc}): {(stderr or '')[-800:]}")
    return _load_lab(cache)


def _load_lab(path: Path) -> list[tuple[float, float, str]]:
    out = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 3:
            continue
        try:
            out.append((float(p[0]), float(p[1]), p[2]))
        except ValueError:
            continue
    return out


def bass_pc_per_segment(labels: list[tuple[float, float, str]],
                        seg_bounds: list[tuple[float, float]]) -> np.ndarray:
    """Music-x-lab sounding-bass pc for each [t0,t1) segment (midpoint lookup).

    Returns int array; entry is -1 where music-x-lab has no chord (N) or no
    overlapping segment at the midpoint.
    """
    out = np.full(len(seg_bounds), -1, dtype=np.int64)
    for i, (a, b) in enumerate(seg_bounds):
        t = 0.5 * (a + b)
        for t0, t1, lab in labels:
            if t0 <= t < t1:
                pc = bass_pc_of_label(lab)
                if pc is not None:
                    out[i] = pc
                break
    return out


def quality_sev_of_label(label: str) -> str | None:
    """Pipeline sev_h token for a music-x-lab Harte label, or None for N/X.

    'C:maj7/E' -> 'maj7'; 'D:min' -> 'min'; 'G:sus4(b7)' -> '7sus4'.  Maps the
    music-x-lab submission vocabulary to the quality spellings the chart adapter
    (render_youtube_chart._QUALITY_TO_IREAL) understands.  Unknown qualities
    degrade to 'maj' (they don't occur in the submission dict).
    """
    if not label:
        return None
    lab = label.strip()
    if lab in ("N", "X", ""):
        return None
    body = lab.split("/", 1)[0]           # drop bass slash
    q = body.split(":", 1)[1] if ":" in body else "maj"
    return _MUSX_Q_TO_SEV.get(q, "maj")


def root_quality_per_segment(
    labels: list[tuple[float, float, str]],
    seg_bounds: list[tuple[float, float]],
) -> list[tuple[int, str | None]]:
    """music-x-lab (root_pc, sev_h) for each [t0,t1) segment (midpoint lookup).

    Entry is ``(-1, None)`` where music-x-lab has no chord (N) or no overlapping
    segment at the midpoint — the caller falls back to the NNLS-24 heads there.
    Root/quality come straight from music-x-lab's own prediction, which beats the
    in-house NNLS-24 heads by a large margin on RWC (docs/known_issues.md FAIR
    bake-off, 2026-07-17): +7.3pp root / +13.5pp quality / +13.9pp joint.
    """
    out: list[tuple[int, str | None]] = []
    for a, b in seg_bounds:
        t = 0.5 * (a + b)
        picked = (-1, None)
        for t0, t1, lab in labels:
            if t0 <= t < t1:
                r = _parse_root(lab.split("/", 1)[0].split(":", 1)[0])
                q = quality_sev_of_label(lab)
                if r is not None and q is not None:
                    picked = (r, q)
                break
        out.append(picked)
    return out


def no_chord_per_segment(
    labels: list[tuple[float, float, str]],
    seg_bounds: list[tuple[float, float]],
) -> np.ndarray:
    """Boolean mask: True where music-x-lab EXPLICITLY labels the segment N/X.

    Distinguishes a genuine no-chord span (music-x-lab wrote ``N``/``X`` at the
    segment midpoint) from the ``(-1, None)`` "no overlapping segment" case that
    ``root_quality_per_segment`` also returns.  The caller uses this to emit a
    first-class no-chord cell (silence / N.C.) instead of falling back to the
    NNLS-24 heads, which always invent a chord (12-way argmax, no reject option)
    at meaningless displayed confidence on chordless audio (known_issues.md
    2026-07-19 ★ CHORDS / NO-CHORD).

    A segment with no overlapping music-x-lab label at all is left False (musx is
    simply silent there, not asserting no-chord) — the NNLS fallback still runs,
    unchanged.
    """
    out = np.zeros(len(seg_bounds), dtype=bool)
    for i, (a, b) in enumerate(seg_bounds):
        t = 0.5 * (a + b)
        for t0, t1, lab in labels:
            if t0 <= t < t1:
                out[i] = lab.strip() in ("N", "X", "")
                break
    return out


def routed_bass_pc(musx_bass: int, nnls_bass: int, root: int) -> int:
    """Validated 'NNLS root-veto' routing (rule F, docs/known_issues.md).

    music-x-lab is primary; NNLS-24 argmax vetoes a music-x-lab inversion claim
    ONLY when NNLS itself reads root-position (kills music-x-lab's false-positive
    inversions).  On the full 100-song RWC set: 0.9196 vs music-x-lab-alone
    0.8999 (+2.0pp overall, +2.4pp root-position, −1.2pp inversions).  If
    music-x-lab is unavailable for this segment (musx_bass < 0), defer to NNLS.
    """
    if musx_bass < 0:
        return nnls_bass
    if musx_bass != nnls_bass and nnls_bass == root:
        return nnls_bass
    return musx_bass
